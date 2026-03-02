"""
Unified fitting module for OLS estimation and standard errors.

Provides two approaches:
- NumpyFitter: In-memory weighted least squares using numpy
- DuckDBFitter: Out-of-core estimation using DuckDB sufficient statistics

Both fitters implement the same interface for seamless switching based on data size.

Canonical Parameter Names (v0.6.0+):
-------------------------------------
All fitters use these standardized parameter names:

Data parameters:
- X: np.ndarray - Design matrix (n, k)
- y: np.ndarray - Response vector (n,)
- weights: np.ndarray - Frequency weights (n,)
- coefficients: Optional[np.ndarray] - Pre-computed coefficients
- residuals: Optional[np.ndarray] - Pre-computed residuals

Variance-covariance parameters:
- vcov_type: str - Standard error type ("iid", "HC1", "HC2", "HC3")
- ssc_dict: Optional[Dict[str, Any]] - Small sample correction config
- k_fe: int - Number of fixed effect levels
- n_fe: int - Number of fixed effect variables

Clustering (Numpy):
- cluster_ids: Optional[np.ndarray] - Cluster identifiers (n,)

Clustering (DuckDB):
- cluster_col: Optional[str] - Cluster column name

Instrumental Variables (Numpy):
- Z: Optional[np.ndarray] - Instrument matrix (n, m)
- is_iv: bool - Whether this is IV regression

Instrumental Variables (DuckDB):
- z_cols: Optional[List[str]] - Instrument column names
- is_iv: bool - Whether this is IV regression
"""

import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

import duckdb

from .linalg import safe_solve, safe_inv
from .vcov import (
    SSCConfig,
    VcovContext,
    VcovSpec,
    compute_iid_vcov,
    compute_hetero_vcov,
    compute_cluster_vcov,
)
from . import (
    compute_sufficient_stats_numpy,
    compute_sufficient_stats_sql,
    compute_residual_aggregates_numpy,
    compute_residual_aggregates_sql
)

logger = logging.getLogger(__name__)

# Constants
DEFAULT_ALPHA = 1e-8
CONDITION_NUMBER_THRESHOLD = 1e12


# ============================================================================
# Fitter Result Container (Dataclass)
# ============================================================================

@dataclass
class FitterResult:
    """Container for estimation results from any fitter.
    
    Includes optional residual statistics to enable efficient vcov 
    recomputation without re-running estimation.
    """
    coefficients: np.ndarray
    coef_names: List[str]
    n_obs: int
    vcov: Optional[np.ndarray] = None
    se_type: str = "none"
    r_squared: Optional[float] = None
    rss: Optional[float] = None
    XtX: Optional[np.ndarray] = None
    Xty: Optional[np.ndarray] = None
    n_clusters: Optional[int] = None
    vcov_meta: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
    
    # Residual statistics for vcov recomputation
    XtX_inv: Optional[np.ndarray] = None
    meat: Optional[np.ndarray] = None
    scores: Optional[np.ndarray] = None
    cluster_scores: Optional[np.ndarray] = None
    leverages: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    
    @property
    def standard_errors(self) -> Optional[np.ndarray]:
        """Compute standard errors from vcov if available."""
        if self.vcov is None:
            return None
        return np.sqrt(np.maximum(np.diag(self.vcov), 1e-16))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        result = {
            "coefficients": self.coefficients,
            "coef_names": self.coef_names,
            "n_obs": self.n_obs,
            "se_type": self.se_type,
        }
        
        # Add optional fields
        if self.vcov is not None:
            result["vcov"] = self.vcov
            result["standard_errors"] = self.standard_errors
        if self.r_squared is not None:
            result["r_squared"] = self.r_squared
        if self.rss is not None:
            result["rss"] = self.rss
        if self.XtX is not None:
            result["XtX"] = self.XtX
        if self.Xty is not None:
            result["Xty"] = self.Xty
        if self.n_clusters is not None:
            result["n_clusters"] = self.n_clusters
        if self.vcov_meta:
            result["vcov_meta"] = self.vcov_meta
        
        # Residual statistics
        for attr in ['XtX_inv', 'meat', 'scores', 'cluster_scores', 'leverages', 'residuals']:
            val = getattr(self, attr)
            if val is not None:
                result[attr] = val
        
        result.update(self.extra)
        return result


# ============================================================================
# Helper Functions
# ============================================================================

def _validate_and_prepare_data(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Validate and prepare input arrays.
    
    Returns:
        Tuple of (X, y, weights, n_rows, n_obs)
    """
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    weights = weights.flatten()
    
    n_rows = X.shape[0]
    n_obs = int(weights.sum())
    
    return X, y, weights, n_rows, n_obs


def _compute_weighted_matrices(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute X'WX and X'Wy with regularization.
    
    Returns:
        Tuple of (XtX, Xty)
    """
    sqrt_w = np.sqrt(weights).reshape(-1, 1)
    Xw = X * sqrt_w
    yw = y * sqrt_w
    
    XtX = Xw.T @ Xw + alpha * np.eye(X.shape[1])
    Xty = Xw.T @ yw
    
    return XtX, Xty


def compute_vcov_dispatch(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    coefficients: np.ndarray,
    residuals: Optional[np.ndarray],
    XtX_inv: np.ndarray = None,
    vcov_spec: VcovSpec = None,
    cluster_ids: Optional[np.ndarray] = None,
    k_fe: int = 0,
    n_fe: int = 0,
    k_fe_nested: int = 0,
    n_fe_fully_nested: int = 0,
    Z: Optional[np.ndarray] = None,
    is_iv: bool = False,
    alpha: float = DEFAULT_ALPHA,
    # Convenience aliases
    XtXinv: np.ndarray = None,
    vcov_type: Optional[str] = None,
    ssc_dict: Optional[Dict[str, Any]] = None,
    kfe: Optional[int] = None,
    nfe: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """Unified variance-covariance computation dispatcher.
    
    Parameters
    ----------
    X : np.ndarray
        Design matrix (n, k)
    y : np.ndarray
        Response vector (n,)
    weights : np.ndarray
        Frequency weights (n,)
    coefficients : np.ndarray
        Coefficient estimates (k,)
    residuals : np.ndarray, optional
        Pre-computed residuals. If None, computed as y - X @ coefficients
    XtX_inv : np.ndarray
        (X'X)^{-1} matrix (k, k)
    vcov_spec : VcovSpec
        Fully-parsed vcov specification (type, SSC config, cluster info)
    cluster_ids : np.ndarray, optional
        Cluster identifiers (n,)
    k_fe : int
        Number of fixed effect levels
    n_fe : int
        Number of fixed effect variables
    Z : np.ndarray, optional
        Instrument matrix (n, m) for IV
    is_iv : bool
        Whether this is IV regression
    alpha : float
        Regularization parameter for IV matrices
        
    Returns
    -------
    Tuple of (vcov, vcov_meta, aggregates)
    """
    # Handle aliases
    if XtXinv is not None and XtX_inv is None:
        XtX_inv = XtXinv
    if kfe is not None:
        k_fe = kfe
    if nfe is not None:
        n_fe = nfe
    # Build vcov_spec from vcov_type/ssc_dict if not provided directly
    if vcov_spec is None:
        _vtype = vcov_type or 'HC1'
        # If cluster_ids provided and vcov_type is not explicitly CRV, use CRV1
        if cluster_ids is not None and _vtype not in ('CRV1', 'CRV3'):
            vcov_spec = VcovSpec.build('CRV1', ssc_dict)
            vcov_spec = VcovSpec(
                vcov_type='CRV',
                vcov_detail='CRV1',
                is_clustered=True,
                cluster_vars=None,
                ssc=vcov_spec.ssc,
            )
        else:
            vcov_spec = VcovSpec.build(_vtype, ssc_dict)

    n_rows, n_features = X.shape
    n_obs = int(weights.sum())
    theta = coefficients.flatten()
    
    # Compute residuals if not provided
    if residuals is None:
        residuals = y.flatten() - X @ theta
    else:
        residuals = residuals.flatten()
    
    aggregates = {
        'XtX_inv': XtX_inv,
        'residuals': residuals,
        'theta': theta
    }
    
    # Compute IV matrices if needed
    tXZ = None
    tZZinv = None
    tZX = None
    if is_iv and Z is not None:
        sqrt_w = np.sqrt(weights).reshape(-1, 1)
        Xw = X * sqrt_w
        Zw = Z * sqrt_w
        
        tXZ = Xw.T @ Zw
        tZZ = Zw.T @ Zw + alpha * np.eye(Z.shape[1])
        tZZinv = safe_inv(tZZ, use_pinv=True)
        tZX = tXZ.T
        
        aggregates.update({'tXZ': tXZ, 'tZZinv': tZZinv, 'tZX': tZX})
    
    # Dispatch based on vcov type
    if vcov_spec.vcov_type == 'CRV':
        # Cluster-robust
        agg = compute_residual_aggregates_numpy(
            theta=theta,
            X=X,
            y=y,
            weights=weights,
            residuals=residuals,
            cluster_ids=cluster_ids,
            compute_cluster_scores=True,
            Z=Z,
            is_iv=is_iv
        )
        
        context = VcovContext(
            N=n_obs, k=n_features, kfe=k_fe, nfe=n_fe,
            kfenested=k_fe_nested, nfefullynested=n_fe_fully_nested
        )
        vcov, vcov_meta = compute_cluster_vcov(
            bread=XtX_inv,
            cluster_scores=agg['cluster_scores'],
            context=context,
            G=agg['n_clusters'],
            ssc_config=vcov_spec.ssc,
            is_iv=is_iv,
            tXZ=tXZ,
            tZZinv=tZZinv,
            tZX=tZX
        )
        aggregates['cluster_scores'] = agg['cluster_scores']
        aggregates['n_clusters'] = agg['n_clusters']
    
    elif vcov_spec.vcov_type == 'iid':
        # Classical IID
        agg = compute_residual_aggregates_numpy(
            theta=theta,
            X=X,
            y=y,
            weights=weights,
            residuals=residuals,
            compute_rss=True,
            Z=Z,
            is_iv=is_iv
        )
        
        context = VcovContext(
            N=n_obs, k=n_features, kfe=k_fe, nfe=n_fe,
            kfenested=k_fe_nested, nfefullynested=n_fe_fully_nested
        )
        vcov, vcov_meta = compute_iid_vcov(
            bread=XtX_inv,
            rss=agg['rss'],
            context=context,
            ssc_config=vcov_spec.ssc,
            is_iv=is_iv,
            tXZ=tXZ,
            tZZinv=tZZinv,
            tZX=tZX
        )
        aggregates['rss'] = agg['rss']
    
    else:
        # Heteroskedastic (HC1, HC2, HC3)
        compute_lev = vcov_spec.vcov_detail in ["HC2", "HC3"]
        if compute_lev:
            logger.warning(
                f"{vcov_spec.vcov_detail} with compressed data: leverages at stratum level. "
                f"Approximation when strata have multiple observations."
            )
        
        agg = compute_residual_aggregates_numpy(
            theta=theta,
            X=X,
            y=y,
            weights=weights,
            residuals=residuals,
            XtX_inv=XtX_inv if compute_lev else None,
            compute_meat=True,
            compute_leverages=compute_lev,
            Z=Z,
            is_iv=is_iv
        )
        
        vcov, vcov_meta = compute_hetero_vcov(
            bread=XtX_inv,
            meat=agg['meat'],
            leverages=agg.get('leverages'),
            vcov_type_detail=vcov_spec.vcov_detail,
            ssc_config=vcov_spec.ssc,
            N=n_obs,
            k=n_features,
            kfe=k_fe,
            nfe=n_fe,
            k_fe_nested=k_fe_nested,
            n_fe_fully_nested=n_fe_fully_nested,
            is_iv=is_iv,
            tXZ=tXZ,
            tZZinv=tZZinv,
            tZX=tZX
        )
        aggregates['meat'] = agg['meat']
        if 'leverages' in agg:
            aggregates['leverages'] = agg['leverages']
    
    return vcov, vcov_meta, aggregates


# ============================================================================
# Abstract Base Fitter
# ============================================================================

class BaseFitter(ABC):
    """Abstract base class for OLS fitters.
    
    Canonical Interface:
    --------------------
    All fitters accept these standardized parameters:
    
    - X, y, weights: Core data (always frequency weights)
    - coefficients, residuals: Optional pre-computed values
    - vcov_type: Standard error type ("iid", "HC1", "HC2", "HC3")
    - ssc_dict: Small sample correction configuration
    - k_fe, n_fe: Fixed effects parameters
    - cluster_ids (numpy) / cluster_col (duckdb): Clustering
    - Z (numpy) / z_cols (duckdb): Instruments for IV
    - is_iv: Flag for IV estimation
    """
    
    def __init__(self, alpha: float = DEFAULT_ALPHA, se_type: str = "stata"):
        self.alpha = alpha
        self.se_type = se_type
    
    @abstractmethod
    def fit(self, **kwargs) -> FitterResult:
        """Fit the model and return results."""
        pass


# ============================================================================
# Numpy Fitter (In-Memory)
# ============================================================================

class NumpyFitter(BaseFitter):
    """In-memory OLS estimation using numpy."""
    
    def __init__(self, alpha: float = DEFAULT_ALPHA, se_type: str = "stata"):
        super().__init__(alpha=alpha, se_type=se_type)
        self._last_result: Optional[FitterResult] = None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        coef_names: Optional[List[str]] = None,
        cluster_ids: Optional[np.ndarray] = None,
        residuals: Optional[np.ndarray] = None,
        coefficients: Optional[np.ndarray] = None,
        k_fe: int = 0,
        n_fe: int = 0,
    ) -> FitterResult:
        """Fit WLS model using numpy."""
        # Validate and prepare
        X, y, weights, n_rows, n_obs = _validate_and_prepare_data(X, y, weights)
        n_features = X.shape[1]
        
        # Compute sufficient statistics
        XtX, Xty, n_obs, sum_y, sum_y_sq, coef_names_suffstats = compute_sufficient_stats_numpy(
            X=X,
            y=y,
            weights=weights,
            coef_names=coef_names,
            alpha=self.alpha
        )
        
        # Use provided coef_names or fall back to suffstats
        if coef_names is None:
            coef_names = coef_names_suffstats
        
        # Get or compute coefficients
        if coefficients is not None:
            theta = coefficients.flatten()
        else:
            theta = safe_solve(XtX, Xty.flatten(), self.alpha)
        
        # Compute R-squared from suffstats results
        rss = sum_y_sq - theta @ Xty
        mean_y = sum_y / n_obs
        tss = sum_y_sq - n_obs * (mean_y ** 2)
        r_squared = max(0.0, 1.0 - rss / tss) if tss > 0 else 0.0
        
        result = FitterResult(
            coefficients=theta,
            coef_names=coef_names,
            n_obs=n_obs,
            vcov=None,
            se_type="none",
            r_squared=r_squared,
            rss=rss,
            XtX=XtX,
            Xty=Xty.flatten(),
            n_clusters=None,
            vcov_meta={}
        )
        
        self._last_result = result
        return result
    
    def fit_vcov(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        coefficients: Optional[np.ndarray] = None,
        coef_names: Optional[List[str]] = None,
        cluster_ids: Optional[np.ndarray] = None,
        vcov_spec: VcovSpec = None,
        k_fe: int = 0,
        n_fe: int = 0,
        k_fe_nested: int = 0,
        n_fe_fully_nested: int = 0,
        existing_result: Optional[FitterResult] = None,
        Z: Optional[np.ndarray] = None,
        is_iv: bool = False,
        residual_X: Optional[np.ndarray] = None,
        # Convenience aliases
        vcov_type: Optional[str] = None,
        ssc_dict: Optional[Dict[str, Any]] = None,
        kfe: Optional[int] = None,
        nfe: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """Compute variance-covariance matrix.
        
        Parameters
        ----------
        vcov_spec : VcovSpec
            Fully-parsed vcov specification. If None, built from vcov_type/ssc_dict or defaults to HC1.
        vcov_type : str, optional
            SE type string (e.g. 'iid', 'HC1', 'CRV1'). Used to build vcov_spec if not provided.
        ssc_dict : dict, optional
            SSC configuration dict. Used with vcov_type to build vcov_spec.
        kfe : int, optional
            Alias for k_fe (number of FE levels).
        nfe : int, optional
            Alias for n_fe (number of FE variables).
        residual_X : np.ndarray, optional
            Alternative X matrix for residual computation (for 2SLS).
        """
        # Handle aliases
        if kfe is not None:
            k_fe = kfe
        if nfe is not None:
            n_fe = nfe
        if vcov_spec is None:
            _vtype = vcov_type or 'HC1'
            if cluster_ids is not None and _vtype not in ('CRV1', 'CRV3'):
                _base = VcovSpec.build('CRV1', ssc_dict)
                vcov_spec = VcovSpec(
                    vcov_type='CRV',
                    vcov_detail='CRV1',
                    is_clustered=True,
                    cluster_vars=None,
                    ssc=_base.ssc,
                )
            else:
                vcov_spec = VcovSpec.build(_vtype, ssc_dict)

        result = existing_result or self._last_result
        
        # Validate and prepare
        X, y, weights, n_rows, n_obs = _validate_and_prepare_data(X, y, weights)
        n_features = X.shape[1]
        
        # Get or compute coefficients
        if coefficients is None:
            if result and result.coefficients is not None:
                theta = result.coefficients.flatten()
            else:
                raise ValueError("coefficients must be provided or available in existing_result")
        else:
            theta = coefficients.flatten()
        
        # Get or compute XtX_inv (always from main X, not residual_X)
        if result and result.XtX_inv is not None:
            XtX_inv = result.XtX_inv
        else:
            if result and result.XtX is not None:
                XtX = result.XtX
            else:
                XtX, _ = _compute_weighted_matrices(X, y, weights, self.alpha)
            XtX_inv = safe_inv(XtX, use_pinv=True)
        
        # Get or compute residuals
        # For 2SLS: use residual_X if provided (actual endogenous)
        # For OLS: residual_X is None, so uses main X
        if residual_X is not None:
            # Compute residuals using alternative X matrix (2SLS case)
            residuals = y.flatten() - residual_X @ theta
        else:
            residuals = result.residuals if result and result.residuals is not None else None
        
        # Call unified dispatcher
        return compute_vcov_dispatch(
            X=X,
            y=y,
            weights=weights,
            coefficients=theta,
            residuals=residuals,
            XtX_inv=XtX_inv,
            vcov_spec=vcov_spec,
            cluster_ids=cluster_ids,
            k_fe=k_fe,
            n_fe=n_fe,
            k_fe_nested=k_fe_nested,
            n_fe_fully_nested=n_fe_fully_nested,
            Z=Z,
            is_iv=is_iv,
            alpha=self.alpha
        )


# ============================================================================
# DuckDB Fitter (Out-of-Core)
# ============================================================================

class DuckDBFitter(BaseFitter):
    """Out-of-core OLS estimation using DuckDB sufficient statistics."""
    
    def __init__(self, conn: duckdb.DuckDBPyConnection, alpha: float = DEFAULT_ALPHA,
                 se_type: str = "stata"):
        super().__init__(alpha=alpha, se_type=se_type)
        self.conn = conn
        self._last_result: Optional[FitterResult] = None
    
    def fit(
        self,
        table_name: str,
        x_cols: List[str] = None,
        y_col: str = None,
        weight_col: str = "count",
        add_intercept: bool = True,
        cluster_col: Optional[str] = None,
        coefficients: Optional[np.ndarray] = None,
        residual_x_cols: Optional[List[str]] = None,
        # Aliases
        xcols: List[str] = None,
        ycol: str = None,
        weightcol: str = None,
    ) -> FitterResult:
        """Fit WLS model using DuckDB sufficient statistics."""
        # Handle aliases
        if xcols is not None and x_cols is None:
            x_cols = xcols
        if ycol is not None and y_col is None:
            y_col = ycol
        if weightcol is not None:
            weight_col = weightcol
        # Compute sufficient statistics
        # Try to use exact sum_y_sq if available:
        # - For compressed data: sum_{outcome}_sq (created during compression)
        # - For uncompressed data: {outcome}_sq (created in view if needed)
        sum_y_sq_col = f"{y_col}_sq"
        
        XtX, Xty, n_obs, sum_y, sum_y_sq, coef_names = compute_sufficient_stats_sql(
            conn=self.conn,
            table_name=table_name,
            x_cols=x_cols,
            y_col=y_col,
            weight_col=weight_col,
            add_intercept=add_intercept,
            alpha=self.alpha,
            sum_y_sq_col=sum_y_sq_col
        )
        
        n_features = len(coef_names)
        
        # Get or compute coefficients
        theta = coefficients.flatten() if coefficients is not None else safe_solve(XtX, Xty, self.alpha)
        
        # Compute R-squared
        rss = sum_y_sq - theta @ Xty
        mean_y = sum_y / n_obs
        tss = sum_y_sq - n_obs * (mean_y ** 2)
        r_squared = max(0.0, 1.0 - rss / tss) if tss > 0 else 0.0
        
        result = FitterResult(
            coefficients=theta,
            coef_names=coef_names,
            n_obs=n_obs,
            vcov=None,
            se_type="none",
            r_squared=r_squared,
            rss=rss,
            XtX=XtX,
            Xty=Xty,
            n_clusters=None,
            vcov_meta={}
        )
        
        self._last_result = result
        return result
    
    def fit_vcov(
        self,
        table_name: str,
        x_cols: List[str] = None,
        y_col: str = None,
        weight_col: str = "count",
        add_intercept: bool = True,
        coefficients: Optional[np.ndarray] = None,
        cluster_col: Optional[str] = None,
        vcov_spec: VcovSpec = None,
        residual_x_cols: Optional[List[str]] = None,
        k_fe: int = 0,
        n_fe: int = 0,
        k_fe_nested: int = 0,
        n_fe_fully_nested: int = 0,
        existing_result: Optional[FitterResult] = None,
        z_cols: Optional[List[str]] = None,
        is_iv: bool = False,
        # Aliases
        xcols: List[str] = None,
        ycol: str = None,
        weightcol: str = None,
        vcov_type: Optional[str] = None,
        ssc_dict: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """Compute variance-covariance matrix using DuckDB SQL."""
        # Handle aliases
        if xcols is not None and x_cols is None:
            x_cols = xcols
        if ycol is not None and y_col is None:
            y_col = ycol
        if weightcol is not None:
            weight_col = weightcol
        if vcov_spec is None:
            _vtype = vcov_type or 'HC1'
            if cluster_col and _vtype not in ('CRV1', 'CRV3'):
                _base = VcovSpec.build('CRV1', ssc_dict)
                vcov_spec = VcovSpec(
                    vcov_type='CRV',
                    vcov_detail='CRV1',
                    is_clustered=True,
                    cluster_vars=None,
                    ssc=_base.ssc,
                )
            else:
                vcov_spec = VcovSpec.build(_vtype, ssc_dict)

        result = existing_result or self._last_result
        
        # Get or compute sufficient statistics
        if result and result.XtX is not None:
            XtX = result.XtX
            Xty = result.Xty
            n_obs = result.n_obs
            coef_names = result.coef_names
        else:
            # Try to use exact sum_y_sq if available (will check if column exists)
            sum_y_sq_col = f"{y_col}_sq"
            
            XtX, Xty, n_obs, _, _, coef_names = compute_sufficient_stats_sql(
                conn=self.conn,
                table_name=table_name,
                x_cols=x_cols,
                y_col=y_col,
                weight_col=weight_col,
                add_intercept=add_intercept,
                alpha=self.alpha,
                sum_y_sq_col=sum_y_sq_col
            )
        
        n_features = XtX.shape[0]
        
        # Get or compute coefficients
        theta = (coefficients.flatten() if coefficients is not None
                else result.coefficients.flatten() if result and result.coefficients is not None
                else safe_solve(XtX, Xty, self.alpha))
        
        # Get or compute XtX_inv
        XtX_inv = (result.XtX_inv if result and result.XtX_inv is not None
                  else safe_inv(XtX, use_pinv=True))
        
        aggregates = {'XtX_inv': XtX_inv, 'theta': theta}
        
        # Compute IV matrices if needed
        tXZ = None
        tZZinv = None
        tZX = None
        if is_iv and z_cols is not None:
            from .sql_builders import compute_cross_sufficient_stats_sql

            iv_stats = compute_cross_sufficient_stats_sql(
                conn=self.conn,
                table_name=table_name,
                x_cols=x_cols,
                z_cols=z_cols,
                weight_col=weight_col,
                add_intercept=add_intercept
            )
            
            tXZ = iv_stats['tXZ']
            tZZ = iv_stats['tZZ'] + self.alpha * np.eye(iv_stats['tZZ'].shape[0])
            tZZinv = safe_inv(tZZ, use_pinv=True)
            tZX = tXZ.T
            
            aggregates.update({'tXZ': tXZ, 'tZZinv': tZZinv, 'tZX': tZX})
        
        # Dispatch based on vcov type
        if cluster_col:
            agg = compute_residual_aggregates_sql(
                theta=theta,
                conn=self.conn,
                table_name=table_name,
                x_cols=x_cols,
                y_col=y_col,
                weight_col=weight_col,
                cluster_col=cluster_col,
                add_intercept=add_intercept,
                residual_x_cols=residual_x_cols,
                compute_cluster_scores=True,
                z_cols=z_cols,
                is_iv=is_iv
            )
            
            context = VcovContext(
                N=n_obs, k=n_features, kfe=k_fe, nfe=n_fe,
                kfenested=k_fe_nested, nfefullynested=n_fe_fully_nested
            )
            vcov, vcov_meta = compute_cluster_vcov(
                bread=XtX_inv,
                cluster_scores=agg['cluster_scores'],
                context=context,
                G=agg['n_clusters'],
                ssc_config=vcov_spec.ssc,
                is_iv=is_iv,
                tXZ=tXZ,
                tZZinv=tZZinv,
                tZX=tZX
            )
            aggregates.update({'cluster_scores': agg['cluster_scores'], 'n_clusters': agg['n_clusters']})
        
        elif vcov_spec.vcov_type == 'iid':
            rss = (result.rss if result and result.rss is not None
                  else compute_residual_aggregates_sql(
                      theta=theta, conn=self.conn, table_name=table_name,
                      x_cols=x_cols, y_col=y_col, weight_col=weight_col,
                      add_intercept=add_intercept, residual_x_cols=residual_x_cols,
                      compute_rss=True
                  )['rss'])
            
            context = VcovContext(
                N=n_obs, k=n_features, kfe=k_fe, nfe=n_fe,
                kfenested=k_fe_nested, nfefullynested=n_fe_fully_nested
            )
            vcov, vcov_meta = compute_iid_vcov(
                bread=XtX_inv, rss=rss, context=context, ssc_config=vcov_spec.ssc,
                is_iv=is_iv, tXZ=tXZ, tZZinv=tZZinv, tZX=tZX
            )
            aggregates['rss'] = rss
        
        else:
            compute_lev = vcov_spec.vcov_detail in ["HC2", "HC3"]
            if compute_lev:
                logger.warning(
                    f"{vcov_spec.vcov_detail} with compressed data: leverages at stratum level. "
                    f"Approximation when strata have multiple observations."
                )

            # For compressed DuckFE data, y_col is in sum_{outcome} format.
            # The compressed view also stores sum_{outcome}_sq (sum of individual y²
            # per stratum), which lets us compute the exact meat that accounts for
            # within-stratum y variation caused by round_strata compression.
            _y_bare = y_col.strip('"').strip("'")
            _exact_meat_col = f"{_y_bare}_sq"
            try:
                _col_exists = self.conn.execute(
                    f"SELECT column_name FROM (DESCRIBE SELECT * FROM {table_name}) "
                    f"WHERE column_name = '{_exact_meat_col}'"
                ).fetchone()
                sum_y_sq_col_for_meat = _exact_meat_col if _col_exists else None
            except Exception:
                sum_y_sq_col_for_meat = None

            agg = compute_residual_aggregates_sql(
                theta=theta, conn=self.conn, table_name=table_name,
                x_cols=x_cols, y_col=y_col, weight_col=weight_col,
                add_intercept=add_intercept, residual_x_cols=residual_x_cols,
                XtX_inv=XtX_inv if compute_lev else None,
                compute_meat=True, compute_leverages=compute_lev,
                z_cols=z_cols, is_iv=is_iv,
                sum_y_sq_col=sum_y_sq_col_for_meat,
            )
            
            vcov, vcov_meta = compute_hetero_vcov(
                bread=XtX_inv, meat=agg['meat'],
                leverages=agg.get('leverages'),
                vcov_type_detail=vcov_spec.vcov_detail,
                ssc_config=vcov_spec.ssc,
                N=n_obs, k=n_features, k_fe=k_fe, n_fe=n_fe,
                k_fe_nested=k_fe_nested, n_fe_fully_nested=n_fe_fully_nested,
                is_iv=is_iv, tXZ=tXZ, tZZinv=tZZinv, tZX=tZX
            )
            aggregates['meat'] = agg['meat']
            if 'leverages' in agg:
                aggregates['leverages'] = agg['leverages']
        
        return vcov, vcov_meta, aggregates


# ============================================================================
# Factory Function
# ============================================================================

def get_fitter(
    fitter_type: str = "numpy",
    conn: Optional[duckdb.DuckDBPyConnection] = None,
    **kwargs
) -> BaseFitter:
    """Get an instance of a fitter class based on the type."""
    if fitter_type == "numpy":
        return NumpyFitter(**kwargs)
    elif fitter_type == "duckdb":
        if conn is None:
            raise ValueError("Connection object must be provided for DuckDB fitter")
        return DuckDBFitter(conn, **kwargs)
    else:
        raise ValueError(f"Unknown fitter type: {fitter_type}")


# ============================================================================
# Convenience Functions (backward compatible)
# ============================================================================

def wls(X: np.ndarray, y: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Weighted least squares with frequency weights."""
    N = np.sqrt(n).reshape(-1, 1)
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    return np.linalg.lstsq(X * N, y * N, rcond=None)[0]


def ridge_closed_form(X: np.ndarray, y: np.ndarray, n: np.ndarray, lam: float) -> np.ndarray:
    """Ridge regression with data augmented representation."""
    k = X.shape[1]
    N = np.sqrt(n).reshape(-1, 1)
    Xtilde = np.vstack([X * N, np.sqrt(lam) * np.eye(k)])
    ytilde = np.vstack([y * N, np.zeros((k, 1))])
    return np.linalg.lstsq(Xtilde, ytilde, rcond=None)[0]


def ridge_closed_form_batch(X: np.ndarray, y: np.ndarray, n: np.ndarray, lambda_grid: np.ndarray) -> np.ndarray:
    """Optimized ridge regression for multiple lambda values."""
    k = X.shape[1]
    N = np.sqrt(n).reshape(-1, 1)
    Xn, yn = X * N, y * N
    I_k, zeros_k = np.eye(k), np.zeros((k, 1))
    
    coefs = np.zeros((len(lambda_grid), k))
    for i, lam in enumerate(lambda_grid):
        Xtilde = np.vstack([Xn, np.sqrt(lam) * I_k])
        ytilde = np.vstack([yn, zeros_k])
        coefs[i, :] = np.linalg.lstsq(Xtilde, ytilde, rcond=None)[0].flatten()
    
    return coefs


def wls_duckdb(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    x_cols: List[str],
    y_col: str,
    weight_col: str = "count",
    add_intercept: bool = True,
    cluster_col: Optional[str] = None,
    se_type: str = "stata",
    alpha: float = DEFAULT_ALPHA
) -> Dict[str, Any]:
    """Weighted least squares using DuckDB sufficient statistics."""
    fitter = DuckDBFitter(conn=conn, alpha=alpha, se_type=se_type)
    result = fitter.fit(
        table_name=table_name,
        x_cols=x_cols,
        y_col=y_col,
        weight_col=weight_col,
        add_intercept=add_intercept,
        cluster_col=cluster_col
    )
    
    # Compute vcov separately
    vcov, vcov_meta, _ = fitter.fit_vcov(
        table_name=table_name,
        x_cols=x_cols,
        y_col=y_col,
        weight_col=weight_col,
        add_intercept=add_intercept,
        cluster_col=cluster_col,
        coefficients=result.coefficients,
        existing_result=result
    )
    
    result.vcov = vcov
    result.se_type = vcov_meta.get('vcov_type_detail', se_type)
    result.vcov_meta = vcov_meta
    
    return result.to_dict()


# ============================================================================
# Export all public symbols
# ============================================================================

__all__ = [
    'FitterResult',
    'BaseFitter',
    'NumpyFitter',
    'DuckDBFitter',
    'get_fitter',
    'compute_vcov_dispatch',
    'wls',
    'wls_duckdb',
    'ridge_closed_form',
    'ridge_closed_form_batch',
    'DEFAULT_ALPHA',
    'CONDITION_NUMBER_THRESHOLD',
]
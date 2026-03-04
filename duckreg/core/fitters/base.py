"""
Base classes and shared helpers for OLS fitters.

Contains:
- DEFAULT_ALPHA, CONDITION_NUMBER_THRESHOLD
- _resolve_vcov_spec        — centralised VcovSpec builder
- FitterResult              — estimation result container
- _validate_and_prepare_data / _compute_weighted_matrices
- compute_vcov_dispatch     — unified vcov dispatcher
- BaseFitter                — abstract base class
"""

import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

from ..linalg import safe_solve, safe_inv
from ..vcov import (
    VcovContext,
    VcovSpec,
    compute_iid_vcov,
    compute_hetero_vcov,
    compute_cluster_vcov,
)
from .. import (
    compute_sufficient_stats_numpy,
    compute_sufficient_stats_sql,
    compute_residual_aggregates_numpy,
    compute_residual_aggregates_sql,
)
from ..suffstats import SuffStats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ALPHA = 1e-8
CONDITION_NUMBER_THRESHOLD = 1e12


# ---------------------------------------------------------------------------
# Centralised VcovSpec Resolution
# ---------------------------------------------------------------------------

def _resolve_vcov_spec(
    vcov_spec: Optional[VcovSpec],
    vcov_type: Optional[str],
    ssc_dict: Optional[Dict[str, Any]],
    has_clusters: bool,
) -> VcovSpec:
    """Build a VcovSpec from vcov_type/ssc_dict when none is supplied directly.

    Single source of truth — previously duplicated in compute_vcov_dispatch,
    NumpyFitter.fit_vcov, and DuckDBFitter.fit_vcov.
    """
    if vcov_spec is not None:
        return vcov_spec
    _vtype = vcov_type or 'HC1'
    if has_clusters and _vtype not in ('CRV1', 'CRV3'):
        _base = VcovSpec.build('CRV1', ssc_dict)
        return VcovSpec(
            vcov_type='CRV',
            vcov_detail='CRV1',
            is_clustered=True,
            cluster_vars=None,
            ssc=_base.ssc,
        )
    return VcovSpec.build(_vtype, ssc_dict)


# ---------------------------------------------------------------------------
# FitterResult
# ---------------------------------------------------------------------------

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
        for attr in ['XtX_inv', 'meat', 'scores', 'cluster_scores', 'leverages', 'residuals']:
            val = getattr(self, attr)
            if val is not None:
                result[attr] = val
        result.update(self.extra)
        return result


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _validate_and_prepare_data(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Validate and prepare input arrays.

    Returns
    -------
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
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute X'WX and X'Wy with Tikhonov regularization.

    Returns
    -------
    Tuple of (XtX, Xty)
    """
    sqrt_w = np.sqrt(weights).reshape(-1, 1)
    Xw = X * sqrt_w
    yw = y * sqrt_w
    XtX = Xw.T @ Xw + alpha * np.eye(X.shape[1])
    Xty = Xw.T @ yw
    return XtX, Xty


# ---------------------------------------------------------------------------
# Unified vcov dispatcher
# ---------------------------------------------------------------------------

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
    X : np.ndarray  (n, k)
    y : np.ndarray  (n,)
    weights : np.ndarray  frequency weights (n,)
    coefficients : np.ndarray  (k,)
    residuals : np.ndarray, optional  — computed as y - X@theta if None
    XtX_inv : np.ndarray  (k, k)
    vcov_spec : VcovSpec
    cluster_ids : np.ndarray, optional
    k_fe, n_fe : fixed effects parameters
    Z : np.ndarray, optional  instrument matrix (n, m)
    is_iv : bool
    alpha : float  regularization for IV matrices

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

    vcov_spec = _resolve_vcov_spec(vcov_spec, vcov_type, ssc_dict, cluster_ids is not None)

    # Guard against Decimal-typed arrays from DuckDB fetchall() results
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    n_rows, n_features = X.shape
    n_obs = int(weights.sum())
    theta = coefficients.flatten()

    if residuals is None:
        residuals = y.flatten() - X @ theta
    else:
        residuals = residuals.flatten()

    aggregates = {'XtX_inv': XtX_inv, 'residuals': residuals, 'theta': theta}

    # IV matrices
    tXZ = tZZinv = tZX = None
    if is_iv and Z is not None:
        sqrt_w = np.sqrt(weights).reshape(-1, 1)
        Xw, Zw = X * sqrt_w, Z * sqrt_w
        tXZ = Xw.T @ Zw
        tZZ = Zw.T @ Zw + alpha * np.eye(Z.shape[1])
        tZZinv = safe_inv(tZZ, use_pinv=True)
        tZX = tXZ.T
        aggregates.update({'tXZ': tXZ, 'tZZinv': tZZinv, 'tZX': tZX})

    # Dispatch
    if vcov_spec.vcov_type == 'CRV':
        agg = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=weights, residuals=residuals,
            cluster_ids=cluster_ids, compute_cluster_scores=True,
            Z=Z, is_iv=is_iv,
        )
        context = VcovContext(
            N=n_obs, k=n_features, kfe=k_fe, nfe=n_fe,
            kfenested=k_fe_nested, nfefullynested=n_fe_fully_nested,
        )
        vcov, vcov_meta = compute_cluster_vcov(
            bread=XtX_inv, cluster_scores=agg['cluster_scores'],
            context=context, G=agg['n_clusters'], ssc_config=vcov_spec.ssc,
            is_iv=is_iv, tXZ=tXZ, tZZinv=tZZinv, tZX=tZX,
        )
        aggregates['cluster_scores'] = agg['cluster_scores']
        aggregates['n_clusters'] = agg['n_clusters']

    elif vcov_spec.vcov_type == 'iid':
        agg = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=weights, residuals=residuals,
            compute_rss=True, Z=Z, is_iv=is_iv,
        )
        context = VcovContext(
            N=n_obs, k=n_features, kfe=k_fe, nfe=n_fe,
            kfenested=k_fe_nested, nfefullynested=n_fe_fully_nested,
        )
        vcov, vcov_meta = compute_iid_vcov(
            bread=XtX_inv, rss=agg['rss'], context=context,
            ssc_config=vcov_spec.ssc, is_iv=is_iv,
            tXZ=tXZ, tZZinv=tZZinv, tZX=tZX,
        )
        aggregates['rss'] = agg['rss']

    else:
        # Heteroskedastic (HC1 / HC2 / HC3)
        compute_lev = vcov_spec.vcov_detail in ["HC2", "HC3"]
        is_compressed = not np.all(weights == 1)
        if compute_lev and is_compressed:
            logger.warning(
                f"{vcov_spec.vcov_detail} with compressed data: leverages at "
                f"stratum level. Approximation when strata have multiple observations."
            )
        agg = compute_residual_aggregates_numpy(
            theta=theta, X=X, y=y, weights=weights, residuals=residuals,
            XtX_inv=XtX_inv if compute_lev else None,
            compute_meat=True, compute_leverages=compute_lev,
            Z=Z, is_iv=is_iv,
        )
        # For HC2/HC3: recompute meat with per-observation leverage adjustment.
        # compute_residual_aggregates_numpy returns the raw (HC1-style) meat;
        # compute_hetero_vcov ignores leverages when meat is pre-computed.
        # Apply the correction here before the call.
        if compute_lev and 'leverages' in agg:
            h = agg['leverages']
            if vcov_spec.vcov_detail == "HC2":
                adj_scale = 1.0 / np.sqrt(np.maximum(1.0 - h, 1e-10))
            else:  # HC3
                adj_scale = 1.0 / np.maximum(1.0 - h, 1e-10)
            _resid = residuals.flatten() if residuals is not None else (y.flatten() - X @ theta)
            score_x = Z if (is_iv and Z is not None) else X
            adj_scores = score_x * (_resid * np.sqrt(weights) * adj_scale).reshape(-1, 1)
            agg['meat'] = adj_scores.T @ adj_scores
        vcov, vcov_meta = compute_hetero_vcov(
            bread=XtX_inv, meat=agg['meat'],
            leverages=None,  # already incorporated into meat above
            vcov_type_detail=vcov_spec.vcov_detail,
            ssc_config=vcov_spec.ssc,
            N=n_obs, k=n_features, kfe=k_fe, nfe=n_fe,
            k_fe_nested=k_fe_nested, n_fe_fully_nested=n_fe_fully_nested,
            is_iv=is_iv, tXZ=tXZ, tZZinv=tZZinv, tZX=tZX,
        )
        aggregates['meat'] = agg['meat']
        if 'leverages' in agg:
            aggregates['leverages'] = agg['leverages']

    return vcov, vcov_meta, aggregates


# ---------------------------------------------------------------------------
# Abstract Base Fitter
# ---------------------------------------------------------------------------

class BaseFitter(ABC):
    """Abstract base class for OLS fitters.

    Canonical Interface
    -------------------
    - X, y, weights : core data (always frequency weights)
    - coefficients, residuals : optional pre-computed values
    - vcov_type : SE type ("iid", "HC1", "HC2", "HC3")
    - ssc_dict : small sample correction configuration
    - k_fe, n_fe : fixed effects parameters
    - cluster_ids (numpy) / cluster_col (duckdb) : clustering
    - Z (numpy) / z_cols (duckdb) : instruments for IV
    - is_iv : flag for IV estimation
    """

    def __init__(self, alpha: float = DEFAULT_ALPHA, se_type: str = "stata"):
        self.alpha = alpha
        self.se_type = se_type

    # ------------------------------------------------------------------
    # Alias normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_fit_kwargs(**kwargs) -> Dict[str, Any]:
        """Normalise legacy keyword aliases to canonical names.

        Maps: xcols→x_cols, ycol→y_col, weightcol→weight_col,
              kfe→k_fe, nfe→n_fe, XtXinv→XtX_inv.
        """
        mapping = {
            'xcols': 'x_cols',
            'ycol': 'y_col',
            'weightcol': 'weight_col',
            'kfe': 'k_fe',
            'nfe': 'n_fe',
            'XtXinv': 'XtX_inv',
        }
        for old, new in mapping.items():
            if old in kwargs:
                if new not in kwargs:
                    kwargs[new] = kwargs.pop(old)
                else:
                    kwargs.pop(old)
        return kwargs

    # ------------------------------------------------------------------
    # Abstract hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _fetch_suffstats(self, **kwargs) -> SuffStats:
        """Fetch / compute sufficient statistics for estimation."""
        ...

    @abstractmethod
    def _fetch_residual_aggregates(
        self,
        theta: np.ndarray,
        XtX_inv: np.ndarray,
        vcov_spec: VcovSpec,
        n_obs: int,
        n_features: int,
        existing_result: Optional['FitterResult'],
        **kwargs,
    ) -> Dict[str, Any]:
        """Compute residual aggregates (meat, rss, cluster scores, …)."""
        ...

    @abstractmethod
    def fit(self, **kwargs) -> 'FitterResult':
        """Fit the model and return a FitterResult."""
        pass


__all__ = [
    'DEFAULT_ALPHA',
    'CONDITION_NUMBER_THRESHOLD',
    '_resolve_vcov_spec',
    'FitterResult',
    '_validate_and_prepare_data',
    '_compute_weighted_matrices',
    'compute_vcov_dispatch',
    'BaseFitter',
]

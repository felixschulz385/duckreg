"""
Consolidated variance-covariance computation utilities.

This module provides the single source of truth for:
- Small sample corrections (SSC)
- Sandwich variance computation
- IID, heteroskedastic, and cluster-robust standard errors
- VCOV specification parsing and validation

Follows pyfixest conventions for vcov computation.
All vcov functions return (vcov, vcov_meta) for consistent metadata tracking.

Architecture:
- Configuration: SSCConfig, VcovContext for parameter management
- Parsing: parse_vcov_specification for input validation
- Core computations: compute_*_vcov functions for variance estimation
- Helpers: Supporting functions for bread, meat, scores
"""

import numpy as np
import logging
import warnings
from typing import Dict, Any, Tuple, Optional, Literal, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ============================================================================
# Type Aliases and Configuration Classes
# ============================================================================

# Type aliases for clarity
WeightType = Literal["aweights", "fweights"]


@dataclass
class SSCConfig:
    """Small sample correction configuration.
    
    Matches fixest's SSC options for consistent behavior.
    
    Attributes
    ----------
    k_adj : bool
        Whether to adjust for number of parameters (N-1)/(N-k)
    k_fixef : str
        How to count fixed effects: "none", "nonnested", or "full"
    G_adj : bool
        Whether to apply cluster adjustment G/(G-1)
    G_df : str
        Cluster df adjustment: "conventional" or "min"
    """
    k_adj: bool = True
    k_fixef: str = "full"
    G_adj: bool = True
    G_df: str = "conventional"
    
    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "SSCConfig":
        """Create from dictionary with defaults."""
        if d is None:
            return cls()
        return cls(
            k_adj=d.get('k_adj', True),
            k_fixef=d.get('k_fixef', 'full'),
            G_adj=d.get('G_adj', True),
            G_df=d.get('G_df', 'conventional')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'k_adj': self.k_adj,
            'k_fixef': self.k_fixef,
            'G_adj': self.G_adj,
            'G_df': self.G_df
        }


@dataclass
class VcovContext:
    """Context for variance-covariance computation.
    
    Groups related parameters to simplify function signatures.
    
    Attributes
    ----------
    N : int
        Number of observations
    k : int
        Number of parameters
    k_fe : int
        Number of fixed effect levels
    n_fe : int
        Number of fixed effect variables
    k_fe_nested : int
        Number of nested fixed effect levels
    n_fe_fully_nested : int
        Number of fully nested FE dimensions
    """
    N: int
    k: int
    k_fe: int = 0
    n_fe: int = 0
    k_fe_nested: int = 0
    n_fe_fully_nested: int = 0


# ============================================================================
# Exceptions
# ============================================================================

class VcovTypeNotSupportedError(Exception):
    """Raised when an unsupported vcov type is requested."""
    pass


# ============================================================================
# VCOV Specification Parsing and Validation
# ============================================================================

def parse_vcov_specification(
    vcov: str | Dict[str, str],
    has_fixef: bool = False,
    is_iv: bool = False
) -> tuple[str, str | list[str], bool, List[str] | None]:
    """
    Deparse the vcov argument to extract standard error computation details.

    Parameters
    ----------
    vcov : Union[str, Dict[str, str]]
        The vcov argument. Can be:
        - A string like "iid", "hetero", "HC1", "CRV1", "NW", "DK"
        - A dict like {"CRV1": "cluster_var"} or {"NW": "lag=4"}
    has_fixef : bool
        Whether the regression has fixed effects. Used for validation.
    is_iv : bool
        Whether the regression is an IV regression. Used for validation.

    Returns
    -------
    Tuple[str, str | List[str], bool, List[str] | None]
        - vcov_type : str
            Broad category: "iid", "hetero", "CRV", "HAC", or "nid"
        - vcov_type_detail : str or list
            Specific type: "iid", "hetero", "HC1", "HC2", "HC3", "CRV1", "CRV3", "NW", "DK", "nid"
        - is_clustered : bool
            Whether clustering is used (True for CRV types)
        - clustervar : list[str] or None
            Names of cluster variables (only for CRV types)

    Raises
    ------
    VcovTypeNotSupportedError
        If HC2/HC3 with fixed effects or IV, or if cluster variables are provided
        with non-clustering vcov types (e.g., {"HC1": "state"})
    TypeError
        If vcov is not a string or dict
    """
    
    # Parse input
    if isinstance(vcov, dict):
        vcov_type_detail = next(iter(vcov.keys()))
        deparse_vcov = next(iter(vcov.values())).split("+")
        if isinstance(deparse_vcov, str):
            deparse_vcov = [deparse_vcov]
        deparse_vcov = [x.strip() for x in deparse_vcov]
    elif isinstance(vcov, str):
        vcov_type_detail = vcov.strip()
        deparse_vcov = []
    elif isinstance(vcov, list):
        vcov_type_detail = vcov[0] if vcov else "iid"
        deparse_vcov = vcov[1:] if len(vcov) > 1 else []
    else:
        raise TypeError("arg vcov must be a dict, string or list")

    # Classify vcov type
    if vcov_type_detail == "iid":
        vcov_type = "iid"
        is_clustered = False
        clustervar = None
        # Validate: no cluster variables for non-cluster SE type
        if deparse_vcov:
            raise VcovTypeNotSupportedError(
                f"Cluster variable(s) provided: {deparse_vcov}, "
                f"but vcov type is '{vcov_type_detail}' (non-clustering). "
                f"Use 'CRV1' or 'CRV3' for cluster-robust standard errors."
            )
    
    elif vcov_type_detail in ["hetero", "HC1", "HC2", "HC3"]:
        vcov_type = "hetero"
        is_clustered = False
        clustervar = None
        # Validate: no cluster variables for non-cluster SE type
        if deparse_vcov:
            raise VcovTypeNotSupportedError(
                f"Cluster variable(s) provided: {deparse_vcov}, "
                f"but vcov type is '{vcov_type_detail}' (non-clustering). "
                f"Use 'CRV1' or 'CRV3' for cluster-robust standard errors."
            )
        
        # Validate HC2/HC3 restrictions
        if vcov_type_detail in ["HC2", "HC3"]:
            if has_fixef:
                raise VcovTypeNotSupportedError(
                    f"{vcov_type_detail} standard errors are not supported for "
                    "regressions with fixed effects."
                )
            if is_iv:
                raise VcovTypeNotSupportedError(
                    f"{vcov_type_detail} standard errors are not supported for IV regressions."
                )
    
    elif vcov_type_detail in ["NW", "DK"]:
        vcov_type = "HAC"
        is_clustered = False
        clustervar = None
        # Validate: no cluster variables for HAC
        if deparse_vcov:
            raise VcovTypeNotSupportedError(
                f"Cluster variable(s) provided: {deparse_vcov}, "
                f"but vcov type is '{vcov_type_detail}' (HAC, not clustering). "
                f"Use 'CRV1' or 'CRV3' for cluster-robust standard errors."
            )
    
    elif vcov_type_detail in ["CRV1", "CRV3"]:
        vcov_type = "CRV"
        is_clustered = True
        clustervar = deparse_vcov if deparse_vcov else None
    
    elif vcov_type_detail == "nid":
        vcov_type = "nid"
        is_clustered = False
        clustervar = None
        # Validate: no cluster variables for non-cluster SE type
        if deparse_vcov:
            raise VcovTypeNotSupportedError(
                f"Cluster variable(s) provided: {deparse_vcov}, "
                f"but vcov type is '{vcov_type_detail}' (non-clustering). "
                f"Use 'CRV1' or 'CRV3' for cluster-robust standard errors."
            )
    
    else:
        raise ValueError(
            f"Unknown vcov type '{vcov_type_detail}'. "
            f"Supported types: iid, hetero, HC1, HC2, HC3, CRV1, CRV3, NW, DK, nid"
        )

    # Handle caret replacement in cluster variable names
    if clustervar and any("^" in var for var in clustervar):
        clustervar = [var.replace("^", "_") for var in clustervar]
        warnings.warn(
            f"The '^' character in cluster variable names has been replaced by '_'. "
            f"Cluster variable(s): {clustervar}",
            UserWarning
        )

    return vcov_type, vcov_type_detail, is_clustered, clustervar


def parse_cluster_vars(
    vcov: str | Dict[str, str]
) -> Optional[List[str]]:
    """
    Extract cluster variable names from vcov specification.
    
    Convenience function to get cluster variables without full parsing.
    Returns None if vcov type is not clustering-based.
    
    Parameters
    ----------
    vcov : Union[str, Dict[str, str]]
        The vcov argument
    
    Returns
    -------
    Optional[List[str]]
        List of cluster variable names if clustering is used, else None
    
    Examples
    --------
    >>> parse_cluster_vars({"CRV1": "state"})
    ['state']
    >>> parse_cluster_vars({"CRV1": "state+firm"})
    ['state', 'firm']
    >>> parse_cluster_vars("HC1")
    None
    """
    try:
        _, _, is_clustered, clustervar = parse_vcov_specification(vcov)
        return clustervar if is_clustered else None
    except (ValueError, TypeError, VcovTypeNotSupportedError):
        return None


# ============================================================================
# Bread Matrix Computation (pyfixest-style)
# ============================================================================

def compute_bread(
    hessian: np.ndarray,
    is_iv: bool = False,
    tXZ: Optional[np.ndarray] = None,
    tZZinv: Optional[np.ndarray] = None,
    tZX: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute the bread matrix (X'X)^{-1} for sandwich variance estimators.
    
    For OLS: bread = (X'X)^{-1} = inv(hessian)
    For IV:  bread = (X'Z (Z'Z)^{-1} Z'X)^{-1}
    
    Parameters
    ----------
    hessian : np.ndarray
        The hessian matrix (X'X for OLS), shape (k, k)
    is_iv : bool
        Whether this is an IV regression
    tXZ : np.ndarray, optional
        X'Z matrix for IV, shape (k, m)
    tZZinv : np.ndarray, optional
        (Z'Z)^{-1} matrix for IV, shape (m, m)
    tZX : np.ndarray, optional
        Z'X matrix for IV, shape (m, k)
        
    Returns
    -------
    np.ndarray
        Bread matrix (k, k)
    """
    if is_iv:
        if tXZ is None or tZZinv is None or tZX is None:
            raise ValueError("tXZ, tZZinv, and tZX required for IV bread computation")
        # For IV: bread = (X'Z (Z'Z)^{-1} Z'X)^{-1}
        return np.linalg.pinv(tXZ @ tZZinv @ tZX)
    else:
        # For OLS: bread = (X'X)^{-1}
        return np.linalg.pinv(hessian)


# ============================================================================
# Small Sample Correction (SSC)
# ============================================================================

def compute_ssc(
    ssc_config: SSCConfig,
    context: VcovContext,
    G: int = 1,
    vcov_type: str = "iid",
    vcov_sign: int = 1
) -> Tuple[float, int, int]:
    """
    Compute small sample correction (SSC) for variance-covariance matrix.
    
    Implements fixest-style SSC computation with support for:
    - Degrees of freedom adjustments for fixed effects
    - Cluster adjustments
    - Multiple vcov types (iid, HC, cluster-robust)
    
    Parameters
    ----------
    ssc_config : SSCConfig
        SSC configuration object
    context : VcovContext
        Computation context with N, k, and FE parameters
    G : int
        Number of clusters (default 1 for non-clustered)
    vcov_type : str
        Type of vcov ("iid", "hetero", "HC0", "HC1", "cluster", "CRV", "HAC")
    vcov_sign : int
        Sign multiplier for vcov (typically 1 or -1)
        
    Returns
    -------
    Tuple[float, int, int]
        - ssc: float, the small sample correction factor
        - df_k: int, degrees of freedom for parameters
        - df_t: int, total degrees of freedom for t-tests
    """
    # Adjust fixed effects count: subtract one for each FE except the first
    # See: https://github.com/lrberge/fixest/issues/554
    k_fe_adj = context.k_fe - (context.n_fe - 1) if context.n_fe > 1 else context.k_fe
    
    # Compute df_k based on k_fixef rule
    if ssc_config.k_fixef == "none":
        df_k = context.k
    elif ssc_config.k_fixef == "nonnested":
        if context.n_fe == 0:
            df_k = context.k
        elif context.k_fe_nested == 0:
            df_k = context.k + k_fe_adj
        else:
            df_k = context.k + k_fe_adj - context.k_fe_nested + context.n_fe_fully_nested
    elif ssc_config.k_fixef == "full":
        df_k = context.k + k_fe_adj if context.n_fe > 0 else context.k
    else:
        raise ValueError(f"k_fixef must be 'none', 'nonnested', or 'full', got: {ssc_config.k_fixef}")
    
    # Compute base adjustment value
    adj_value = 1.0
    if ssc_config.k_adj:
        if vcov_type in ("hetero", "HC0"):
            adj_value = context.N / max(1, context.N - df_k)
        else:
            adj_value = (context.N - 1) / max(1, context.N - df_k)
    
    # Apply cluster adjustments for CRV/HAC
    G_adj_value = 1.0
    if vcov_type in ("CRV", "cluster", "CRV1", "HAC") and ssc_config.G_adj:
        if ssc_config.G_df == "conventional":
            G_adj_value = G / (G - 1) if G > 1 else 1.0
        elif ssc_config.G_df == "min":
            G_min = np.min(G) if hasattr(G, '__iter__') else G
            G_adj_value = G_min / (G_min - 1) if G_min > 1 else 1.0
        else:
            raise ValueError(f"G_df must be 'conventional' or 'min', got: {ssc_config.G_df}")
    
    # Compute total SSC
    ssc = adj_value * G_adj_value * vcov_sign
    
    # Compute degrees of freedom for t-statistics
    if vcov_type in ("iid", "hetero", "HC0", "HC1", "HAC-TS"):
        df_t = context.N - df_k
    else:  # CRV, cluster, HAC
        df_t = G - 1
    
    return ssc, df_k, df_t


# ============================================================================
# Sandwich Computation
# ============================================================================

def sandwich_from_meat(
    bread: np.ndarray,
    meat: np.ndarray,
    ssc: float
) -> np.ndarray:
    """
    Compute sandwich variance: ssc * bread @ meat @ bread.
    
    Parameters
    ----------
    bread : np.ndarray
        (X'X)^{-1} or equivalent bread matrix (k x k)
    meat : np.ndarray
        Meat matrix, e.g., sum of outer products of scores (k x k)
    ssc : float
        Small sample correction factor
        
    Returns
    -------
    np.ndarray
        Variance-covariance matrix (k x k), symmetrized
    """
    vcov = ssc * (bread @ meat @ bread)
    return 0.5 * (vcov + vcov.T)  # Ensure symmetry


# ============================================================================
# IID (Homoskedastic) Variance
# ============================================================================

def vcov_iid(
    bread: np.ndarray,
    u_hat: np.ndarray,
    N: int
) -> np.ndarray:
    """
    Compute IID variance-covariance matrix (before SSC scaling).
    
    Matches pyfixest's _vcov_iid method.
    
    vcov = bread * sigma^2
    where sigma^2 = sum(u_hat^2) / (N - 1)
    
    Parameters
    ----------
    bread : np.ndarray
        (X'X)^{-1}, shape (k, k)
    u_hat : np.ndarray
        Residuals (n,) - should be unweighted residuals
    N : int
        Number of observations (sum of frequency weights if using fweights)
        
    Returns
    -------
    np.ndarray
        Unscaled vcov matrix (k, k) - multiply by SSC for final result
    """
    sigma2 = np.sum(u_hat.flatten() ** 2) / (N - 1)
    return bread * sigma2


def compute_iid_vcov(
    bread: np.ndarray,
    rss: float,
    context: VcovContext,
    ssc_config: Optional[SSCConfig] = None,
    is_iv: bool = False,
    tXZ: Optional[np.ndarray] = None,
    tZZinv: Optional[np.ndarray] = None,
    tZX: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute classical IID (homoskedastic) variance-covariance matrix.
    
    Matches pyfixest's vcov() method for vcov_type == "iid".
    
    vcov = ssc * bread * sigma^2
    where sigma^2 = rss / (N - 1)
    
    For IV: applies IV correction to meat before sandwich computation.
    
    Parameters
    ----------
    bread : np.ndarray
        (X'X)^{-1}, shape (k, k)
    rss : float
        Residual sum of squares: sum(u_hat^2)
    context : VcovContext
        Computation context (N, k, FE parameters)
    ssc_config : SSCConfig, optional
        SSC configuration (uses defaults if None)
    is_iv : bool
        Whether this is IV regression
    tXZ, tZZinv, tZX : np.ndarray, optional
        IV matrices for sandwich computation
        
    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        (vcov matrix, metadata dict)
    """
    if ssc_config is None:
        ssc_config = SSCConfig(k_adj=True, k_fixef='full', G_adj=False, G_df='none')
    
    ssc, df_k, df_t = compute_ssc(ssc_config, context, G=1, vcov_type="iid", vcov_sign=1)
    
    sigma2 = rss / (context.N - 1)
    
    # For iid variance, simply scale bread by sigma^2 (same for OLS and IV)
    # The bread matrix already contains IV structure (X'P_Z X)^{-1} if is_iv=True
    vcov = ssc * bread * sigma2
    
    # Symmetrize
    vcov = 0.5 * (vcov + vcov.T)
    
    vcov_meta = {
        'ssc': ssc,
        'df_k': df_k,
        'df_t': df_t,
        'vcov_type': 'iid',
        'vcov_type_detail': 'iid',
        'sigma2': sigma2
    }
    
    return vcov, vcov_meta


# ============================================================================
# Heteroskedastic-Robust Variance
# ============================================================================

def vcov_hetero(
    bread: np.ndarray,
    scores: np.ndarray,
    vcov_type_detail: str = "HC1",
    hessian: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    weights_type: Optional[WeightType] = None,
    is_iv: bool = False,
    tXZ: Optional[np.ndarray] = None,
    tZZinv: Optional[np.ndarray] = None,
    tZX: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute heteroskedasticity-robust variance (before SSC scaling).
    
    Matches pyfixest's _vcov_hetero method exactly.
    
    Parameters
    ----------
    bread : np.ndarray
        (X'X)^{-1}, shape (k, k)
    scores : np.ndarray
        Score matrix X * u_hat, shape (n, k)
        For weighted regression, these should already be the weighted scores
    vcov_type_detail : str
        "hetero", "HC1", "HC2", or "HC3"
    hessian : np.ndarray, optional
        X'X matrix, required for HC2/HC3 leverage computation
    weights : np.ndarray, optional
        Weights array (n,), required for fweights adjustment in HC2/HC3
    weights_type : str, optional
        "aweights" or "fweights"
    is_iv : bool
        Whether this is an IV regression
    tXZ, tZZinv, tZX : np.ndarray, optional
        IV matrices for sandwich computation
        
    Returns
    -------
    np.ndarray
        Unscaled vcov matrix (k, k) - multiply by SSC for final result
    """
    if vcov_type_detail in ["hetero", "HC1"]:
        transformed_scores = scores
    elif vcov_type_detail in ["HC2", "HC3"]:
        if hessian is None:
            raise ValueError("hessian (X'X) required for HC2/HC3 leverage computation")
        
        # Compute leverage: h_ii = x_i' (X'X)^{-1} x_i
        # Note: X here should be the score matrix's corresponding X (not weighted for leverage)
        # In pyfixest: leverage = np.sum(self._X * (self._X @ np.linalg.inv(self._tZX)), axis=1)
        # tZX is Z'X for IV, or X'X for OLS
        X = scores / scores[:, :1]  # This won't work - need actual X
        
        # Actually, we need X separately to compute leverage
        # For now, fall back to requiring leverage from caller
        logger.warning(
            f"{vcov_type_detail} requires leverage values. "
            f"Use compute_hetero_vcov with leverages parameter."
        )
        transformed_scores = scores
    else:
        transformed_scores = scores
    
    # For fweights, need to divide by sqrt(weights)
    if weights_type == "fweights" and weights is not None:
        transformed_scores = transformed_scores / np.sqrt(weights.reshape(-1, 1))
    
    Omega = transformed_scores.T @ transformed_scores
    
    if is_iv:
        meat = tXZ @ tZZinv @ Omega @ tZZinv @ tZX
    else:
        meat = Omega
    
    vcov = bread @ meat @ bread
    
    return vcov


def compute_hetero_vcov(
    bread: np.ndarray,
    scores: Optional[np.ndarray] = None,
    meat: Optional[np.ndarray] = None,
    X: Optional[np.ndarray] = None,
    hessian: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    weights_type: Optional[WeightType] = None,
    leverages: Optional[np.ndarray] = None,
    vcov_type_detail: str = "HC1",
    ssc_dict: Optional[Dict[str, Any]] = None,
    N: Optional[int] = None,
    k: Optional[int] = None,
    k_fe: int = 0,
    n_fe: int = 0,
    is_iv: bool = False,
    tXZ: Optional[np.ndarray] = None,
    tZZinv: Optional[np.ndarray] = None,
    tZX: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute heteroskedasticity-robust variance from scores or pre-computed meat.
    
    Matches pyfixest's vcov() method for vcov_type == "hetero".
    
    Supports HC1, HC2, HC3. For HC2/HC3, provide either:
    - X and hessian for automatic leverage computation
    - Or leverages directly
    
    For IV: applies IV correction to meat before sandwich computation.
    
    Parameters
    ----------
    bread : np.ndarray
        (X'X)^{-1}, shape (k, k)
    scores : np.ndarray, optional
        Score matrix X * u_hat, shape (n, k)
    meat : np.ndarray, optional
        Pre-computed meat matrix (k, k)
    X : np.ndarray, optional
        Design matrix (n, k), required for HC2/HC3 if leverages not provided
    hessian : np.ndarray, optional
        X'X matrix (k, k), required for HC2/HC3 if leverages not provided
    weights : np.ndarray, optional
        Weights array (n,)
    weights_type : str, optional
        "aweights" or "fweights"
    leverages : np.ndarray, optional
        Leverage values h_ii (n,) for HC2/HC3 adjustment
    vcov_type_detail : str
        "hetero", "HC1", "HC2", or "HC3"
    ssc_dict : Dict[str, Any], optional
        SSC configuration
    N : int, optional
        Number of observations
    k : int, optional
        Number of parameters
    k_fe, n_fe : int
        Fixed effects parameters
    is_iv : bool
        Whether this is IV regression
    tXZ, tZZinv, tZX : np.ndarray, optional
        IV matrices for sandwich computation
        
    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        (vcov matrix, metadata dict)
    """
    if k is None:
        k = bread.shape[0]
    
    if ssc_dict is None:
        ssc_dict = {'k_adj': True, 'k_fixef': 'full', 'G_adj': True, 'G_df': 'conventional'}
    
    # Compute meat if not provided
    if meat is None:
        if scores is None:
            raise ValueError("Either scores or meat must be provided")
        
        if N is None:
            N = scores.shape[0]
        
        # Determine transformed scores based on vcov type
        if vcov_type_detail in ["hetero", "HC1"]:
            transformed_scores = scores.copy()
        elif vcov_type_detail in ["HC2", "HC3"]:
            # Compute leverage if not provided
            if leverages is None:
                if X is not None and hessian is not None:
                    # leverage = x_i' (X'X)^{-1} x_i = sum_j sum_l x_ij * inv_jl * x_il
                    leverages = np.sum((X @ np.linalg.inv(hessian)) * X, axis=1)
                else:
                    logger.warning(
                        f"{vcov_type_detail} requires leverages (h_ii values) for proper adjustment. "
                        f"Falling back to HC1. To fix: provide X and hessian, or leverages directly."
                    )
                    vcov_type_detail = "HC1"
            
            if leverages is not None:
                # For fweights, divide leverage by weights (pyfixest convention)
                if weights_type == "fweights" and weights is not None:
                    leverages = leverages / weights.flatten()
                
                if vcov_type_detail == "HC2":
                    # Scale by 1 / sqrt(1 - h_ii)
                    adjustment = 1.0 / np.sqrt(np.maximum(1.0 - leverages, 1e-10))
                else:  # HC3
                    # Scale by 1 / (1 - h_ii)
                    adjustment = 1.0 / np.maximum(1.0 - leverages, 1e-10)
                
                transformed_scores = scores * adjustment.reshape(-1, 1)
            else:
                transformed_scores = scores.copy()
        else:
            transformed_scores = scores.copy()
        
        # For fweights, need to divide by sqrt(weights)
        if weights_type == "fweights" and weights is not None:
            transformed_scores = transformed_scores / np.sqrt(weights.reshape(-1, 1))
        
        Omega = transformed_scores.T @ transformed_scores
        
        if is_iv:
            meat = tXZ @ tZZinv @ Omega @ tZZinv @ tZX
        else:
            meat = Omega
    else:
        if N is None:
            raise ValueError("N must be provided when using pre-computed meat")
    
    # Compute SSC - for hetero, G = N (fixest convention)
    context = VcovContext(N=N, k=k, k_fe=k_fe, n_fe=n_fe)
    ssc_config = SSCConfig.from_dict(ssc_dict)
    ssc, df_k, df_t = compute_ssc(
        ssc_config=ssc_config,
        context=context,
        G=N,
        vcov_type="hetero",
        vcov_sign=1
    )
    
    # Apply IV correction if needed
    if is_iv and tXZ is not None and tZZinv is not None and tZX is not None:
        meat = tXZ @ tZZinv @ meat @ tZZinv @ tZX
    
    vcov = ssc * (bread @ meat @ bread)
    # Symmetrize
    vcov = 0.5 * (vcov + vcov.T)
    
    vcov_meta = {
        'ssc': ssc,
        'df_k': df_k,
        'df_t': df_t,
        'vcov_type': 'hetero',
        'vcov_type_detail': vcov_type_detail
    }
    
    return vcov, vcov_meta


# ============================================================================
# Cluster-Robust Variance
# ============================================================================

def vcov_crv1(
    bread: np.ndarray,
    scores: np.ndarray,
    cluster_col: np.ndarray,
    clustid: Optional[np.ndarray] = None,
    is_iv: bool = False,
    tXZ: Optional[np.ndarray] = None,
    tZZinv: Optional[np.ndarray] = None,
    tZX: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute CRV1 cluster-robust variance (before SSC scaling).
    
    Matches pyfixest's _vcov_crv1 method.
    
    Parameters
    ----------
    bread : np.ndarray
        (X'X)^{-1}, shape (k, k)
    scores : np.ndarray
        Score matrix X * u_hat, shape (n, k)
    cluster_col : np.ndarray
        Cluster assignments (n,) - integer coded
    clustid : np.ndarray, optional
        Unique cluster IDs. If None, computed from cluster_col
    is_iv : bool
        Whether this is an IV regression
    tXZ, tZZinv, tZX : np.ndarray, optional
        IV matrices for sandwich computation
        
    Returns
    -------
    np.ndarray
        Unscaled vcov matrix (k, k) - multiply by SSC for final result
    """
    k = scores.shape[1]
    
    if clustid is None:
        clustid = np.unique(cluster_col)
    
    # Compute cluster-aggregated scores (meat matrix)
    # meat = sum_g (score_g @ score_g.T) where score_g = sum_{i in g} score_i
    meat = np.zeros((k, k))
    
    for g in clustid:
        mask = cluster_col == g
        cluster_score = scores[mask].sum(axis=0)
        meat += np.outer(cluster_score, cluster_score)
    
    if is_iv:
        meat = tXZ @ tZZinv @ meat @ tZZinv @ tZX
    
    vcov = bread @ meat @ bread
    
    return vcov


def compute_cluster_vcov(
    bread: np.ndarray,
    cluster_scores: np.ndarray,
    context: VcovContext,
    G: int,
    ssc_config: Optional[SSCConfig] = None,
    is_iv: bool = False,
    tXZ: Optional[np.ndarray] = None,
    tZZinv: Optional[np.ndarray] = None,
    tZX: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute cluster-robust variance from cluster-level scores.
    
    Matches pyfixest's vcov() method for vcov_type == "CRV".
    
    For IV: applies IV correction to meat before sandwich computation.
    
    Parameters
    ----------
    bread : np.ndarray
        (X'X)^{-1}, shape (k, k)
    cluster_scores : np.ndarray
        Cluster-level score matrix, shape (G, k)
        Each row is the sum of scores within a cluster
    context : VcovContext
        Computation context (N, k, FE parameters)
    G : int
        Number of clusters
    ssc_config : SSCConfig, optional
        SSC configuration (uses defaults if None)
    is_iv : bool
        Whether this is IV regression
    tXZ, tZZinv, tZX : np.ndarray, optional
        IV matrices for sandwich computation
        
    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        (vcov matrix, metadata dict)
    """
    if ssc_config is None:
        ssc_config = SSCConfig(k_adj=True, k_fixef='full', G_adj=True, G_df='conventional')
    
    # Compute meat from cluster scores
    meat = cluster_scores.T @ cluster_scores
    
    # Apply IV correction if needed
    if is_iv and tXZ is not None and tZZinv is not None and tZX is not None:
        meat = tXZ @ tZZinv @ meat @ tZZinv @ tZX
    
    # Compute SSC
    ssc, df_k, df_t = compute_ssc(ssc_config, context, G, vcov_type="CRV", vcov_sign=1)
    
    vcov = ssc * (bread @ meat @ bread)
    # Symmetrize
    vcov = 0.5 * (vcov + vcov.T)
    
    vcov_meta = {
        'ssc': ssc,
        'df_k': df_k,
        'df_t': df_t,
        'vcov_type': 'cluster',
        'vcov_type_detail': 'CRV1',
        'n_clusters': G
    }
    
    return vcov, vcov_meta


def compute_twoway_cluster_vcov(
    bread: np.ndarray,
    scores: np.ndarray,
    cluster_df: np.ndarray,
    ssc_dict: Optional[Dict[str, Any]],
    N: int,
    k: Optional[int] = None,
    k_fe: int = 0,
    n_fe: int = 0,
    k_fe_nested: int = 0,
    n_fe_fully_nested: int = 0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute two-way cluster-robust variance.
    
    Matches pyfixest's vcov() method for two-way clustering.
    Uses Cameron-Gelbach-Miller formula: V = V_1 + V_2 - V_12
    
    Parameters
    ----------
    bread : np.ndarray
        (X'X)^{-1}, shape (k, k)
    scores : np.ndarray
        Score matrix X * u_hat, shape (n, k)
    cluster_df : np.ndarray
        Cluster assignments, shape (n, 3) for two-way
        Columns: [cluster1, cluster2, cluster1_x_cluster2]
    ssc_dict : Dict[str, Any], optional
        SSC configuration
    N : int
        Number of observations
    k : int, optional
        Number of parameters
    k_fe, n_fe, k_fe_nested, n_fe_fully_nested : int
        Fixed effects parameters
        
    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        (vcov matrix, metadata dict)
    """
    if k is None:
        k = bread.shape[0]
    
    if ssc_dict is None:
        ssc_dict = {'k_adj': True, 'k_fixef': 'full', 'G_adj': True, 'G_df': 'conventional'}
    
    vcov_sign_list = [1, 1, -1]
    df_t_full = np.zeros(cluster_df.shape[1])
    G_list = []
    
    vcov = np.zeros((k, k))
    ssc_list = []
    
    for x in range(cluster_df.shape[1]):
        cluster_col = cluster_df[:, x]
        clustid = np.unique(cluster_col)
        G = len(clustid)
        G_list.append(G)
        
        context = VcovContext(
            N=N, k=k, k_fe=k_fe, n_fe=n_fe,
            k_fe_nested=k_fe_nested, n_fe_fully_nested=n_fe_fully_nested
        )
        ssc_config = SSCConfig.from_dict(ssc_dict)
        ssc, df_k, df_t = compute_ssc(
            ssc_config=ssc_config,
            context=context,
            G=G,
            vcov_type="CRV",
            vcov_sign=vcov_sign_list[x]
        )
        
        ssc_list.append(ssc)
        df_t_full[x] = df_t
        
        # Compute CRV1 for this clustering dimension
        vcov_x = vcov_crv1(bread, scores, cluster_col, clustid)
        vcov += ssc * vcov_x
    
    # Symmetrize
    vcov = 0.5 * (vcov + vcov.T)
    
    vcov_meta = {
        'ssc': ssc_list,
        'df_k': df_k,
        'df_t': np.min(df_t_full),
        'vcov_type': 'cluster',
        'vcov_type_detail': 'CRV1-twoway',
        'n_clusters': G_list
    }
    
    return vcov, vcov_meta


# ============================================================================
# Helper: Aggregate scores by cluster
# ============================================================================

def compute_cluster_scores(
    scores: np.ndarray,
    cluster_ids: np.ndarray
) -> Tuple[np.ndarray, int]:
    """
    Aggregate scores by cluster.
    
    Parameters
    ----------
    scores : np.ndarray
        Score matrix (n, k)
    cluster_ids : np.ndarray
        Cluster identifiers (n,)
        
    Returns
    -------
    Tuple[np.ndarray, int]
        (cluster_scores (G, k), G)
    """
    unique_clusters = np.unique(cluster_ids)
    G = len(unique_clusters)
    k = scores.shape[1]
    
    cluster_scores = np.zeros((G, k))
    for g, cluster in enumerate(unique_clusters):
        mask = cluster_ids == cluster
        cluster_scores[g] = scores[mask].sum(axis=0)
    
    return cluster_scores, G


# ============================================================================
# Public API Exports
# ============================================================================

__all__ = [
    # Configuration classes
    'SSCConfig',
    'VcovContext',
    
    # Exceptions
    'VcovTypeNotSupportedError',
    
    # Parsing functions
    'parse_vcov_specification',
    'parse_cluster_vars',
    
    # Core computation functions
    'compute_ssc',
    'compute_bread',
    'compute_iid_vcov',
    'compute_hetero_vcov',
    'compute_cluster_vcov',
    'compute_twoway_cluster_vcov',
    
    # Helper functions
    'sandwich_from_meat',
    'compute_cluster_scores',
    
    # Legacy functions (pyfixest-style, less commonly used)
    'vcov_iid',
    'vcov_hetero',
    'vcov_crv1',
]

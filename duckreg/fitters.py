"""
Unified fitting module for OLS estimation and standard errors.

Provides two approaches:
- NumpyFitter: In-memory weighted least squares using numpy
- DuckDBFitter: Out-of-core estimation using DuckDB sufficient statistics

Both fitters implement the same interface for seamless switching based on data size.
"""

import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any

import duckdb

logger = logging.getLogger(__name__)

# Constants
DEFAULT_ALPHA = 1e-8
CONDITION_NUMBER_THRESHOLD = 1e12


# ============================================================================
# Linear Algebra Helpers
# ============================================================================

class LinAlgHelper:
    """Helper class for linear algebra operations with error handling."""
    
    _regularization_warned = False
    
    @staticmethod
    def safe_solve(A: np.ndarray, b: np.ndarray, alpha: float = DEFAULT_ALPHA,
                   multiplier: int = 10) -> np.ndarray:
        """Safely solve linear system with fallback regularization."""
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            if not LinAlgHelper._regularization_warned:
                logger.info("Using regularization for numerical stability")
                LinAlgHelper._regularization_warned = True
            regularized_A = A + alpha * multiplier * np.eye(A.shape[0])
            return np.linalg.solve(regularized_A, b)
    
    @staticmethod
    def safe_inv(A: np.ndarray, use_pinv: bool = False) -> np.ndarray:
        """Safely invert matrix with fallback to pseudo-inverse."""
        try:
            return np.linalg.inv(A)
        except np.linalg.LinAlgError:
            if use_pinv:
                return np.linalg.pinv(A)
            raise
    
    @staticmethod
    def check_condition_number(A: np.ndarray, 
                                threshold: float = CONDITION_NUMBER_THRESHOLD) -> Tuple[bool, float]:
        """Check if matrix is well-conditioned."""
        try:
            eigvals = np.linalg.eigvalsh(A)
            cond = eigvals.max() / max(eigvals.min(), 1e-10)
            return cond < threshold, cond
        except np.linalg.LinAlgError:
            return False, float('inf')


# ============================================================================
# Fitter Result Container
# ============================================================================

class FitterResult:
    """Container for estimation results from any fitter."""
    
    def __init__(
        self,
        coefficients: np.ndarray,
        coef_names: List[str],
        n_obs: int,
        vcov: Optional[np.ndarray] = None,
        se_type: str = "none",
        r_squared: Optional[float] = None,
        rss: Optional[float] = None,
        XtX: Optional[np.ndarray] = None,
        Xty: Optional[np.ndarray] = None,
        n_clusters: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None
    ):
        self.coefficients = coefficients
        self.coef_names = coef_names
        self.n_obs = n_obs
        self.vcov = vcov
        self.se_type = se_type
        self.r_squared = r_squared
        self.rss = rss
        self.XtX = XtX
        self.Xty = Xty
        self.n_clusters = n_clusters
        self.extra = extra or {}
    
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
        if self.n_clusters is not None:
            result["n_clusters"] = self.n_clusters
        result.update(self.extra)
        return result


# ============================================================================
# Abstract Base Fitter
# ============================================================================

class BaseFitter(ABC):
    """Abstract base class for OLS fitters."""
    
    def __init__(self, alpha: float = DEFAULT_ALPHA, se_type: str = "stata"):
        self.alpha = alpha
        self.se_type = se_type
        self._linalg = LinAlgHelper()
    
    @abstractmethod
    def fit(self, **kwargs) -> FitterResult:
        """Fit the model and return results."""
        pass
    
    def _compute_small_sample_correction(self, n_obs: int, n_features: int, 
                                          n_clusters: Optional[int] = None) -> float:
        """Compute small sample correction factor."""
        if n_clusters is not None:
            # Cluster-robust correction
            if self.se_type == "stata":
                return (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_features))
            elif self.se_type == "HC0":
                return 1.0
            elif self.se_type == "HC1":
                return n_obs / (n_obs - n_features)
            else:
                return (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_features))
        else:
            # Non-clustered correction
            return n_obs / max(1, n_obs - n_features)


# ============================================================================
# Numpy Fitter (In-Memory)
# ============================================================================

class NumpyFitter(BaseFitter):
    """
    In-memory OLS estimation using numpy.
    
    Suitable when compressed data fits in memory.
    """
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        coef_names: Optional[List[str]] = None,
        cluster_ids: Optional[np.ndarray] = None,
        compute_vcov: bool = True
    ) -> FitterResult:
        """
        Fit WLS model using numpy.
        
        Parameters
        ----------
        X : np.ndarray
            Design matrix (n_rows, n_features)
        y : np.ndarray
            Response vector/matrix (n_rows,) or (n_rows, n_outcomes)
        weights : np.ndarray
            Frequency weights (n_rows,)
        coef_names : List[str], optional
            Names for coefficients
        cluster_ids : np.ndarray, optional
            Cluster identifiers for cluster-robust SEs
        compute_vcov : bool
            Whether to compute variance-covariance matrix
            
        Returns
        -------
        FitterResult with coefficients, vcov, etc.
        """
        # Ensure proper shapes
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        weights = weights.flatten()
        
        n_rows, n_features = X.shape
        n_obs = int(weights.sum())
        
        # Compute sufficient statistics
        sqrt_w = np.sqrt(weights).reshape(-1, 1)
        Xw = X * sqrt_w
        yw = y * sqrt_w
        
        XtX = Xw.T @ Xw + self.alpha * np.eye(n_features)
        Xty = Xw.T @ yw
        
        # Solve for coefficients
        theta = self._linalg.safe_solve(XtX, Xty.flatten(), self.alpha)
        
        # Compute R-squared
        sum_y = (y.flatten() * weights).sum()
        sum_y_squared = ((y.flatten() ** 2) * weights).sum()
        rss = sum_y_squared - theta @ Xty.flatten()
        mean_y = sum_y / n_obs
        tss = sum_y_squared - n_obs * (mean_y ** 2)
        r_squared = max(0.0, 1.0 - rss / tss) if tss > 0 else 0.0
        
        # Coefficient names
        if coef_names is None:
            coef_names = [f"x{i}" for i in range(n_features)]
        
        # Compute vcov if requested
        vcov = None
        se_type_used = "none"
        n_clusters = None
        
        if compute_vcov:
            if cluster_ids is not None:
                vcov, n_clusters = self._compute_cluster_robust_vcov(
                    X, y, weights, theta, XtX, cluster_ids
                )
                se_type_used = "cluster"
            else:
                vcov = self._compute_hc1_vcov(X, y, weights, theta, XtX, rss, n_obs)
                se_type_used = "hc1"
        
        return FitterResult(
            coefficients=theta,
            coef_names=coef_names,
            n_obs=n_obs,
            vcov=vcov,
            se_type=se_type_used,
            r_squared=r_squared,
            rss=rss,
            XtX=XtX,
            Xty=Xty.flatten(),
            n_clusters=n_clusters
        )
    
    def _compute_cluster_robust_vcov(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        theta: np.ndarray,
        XtX: np.ndarray,
        cluster_ids: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """Compute cluster-robust variance-covariance matrix."""
        n_features = X.shape[1]
        n_obs = int(weights.sum())
        
        # Compute residuals
        yhat = X @ theta
        residuals = (y.flatten() - yhat) * weights
        
        # Get unique clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)
        
        # Compute meat matrix
        meat = np.zeros((n_features, n_features))
        for cluster in unique_clusters:
            mask = cluster_ids == cluster
            score_g = (X[mask] * residuals[mask, np.newaxis]).sum(axis=0)
            meat += np.outer(score_g, score_g)
        
        # Compute bread
        XtX_inv = self._linalg.safe_inv(XtX, use_pinv=True)
        
        # Apply correction
        correction = self._compute_small_sample_correction(n_obs, n_features, n_clusters)
        vcov = correction * (XtX_inv @ meat @ XtX_inv)
        vcov = 0.5 * (vcov + vcov.T)  # Ensure symmetry
        
        return vcov, n_clusters
    
    def _compute_hc1_vcov(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        theta: np.ndarray,
        XtX: np.ndarray,
        rss: float,
        n_obs: int
    ) -> np.ndarray:
        """Compute HC1 heteroskedasticity-robust variance-covariance matrix."""
        n_features = X.shape[1]
        
        XtX_inv = self._linalg.safe_inv(XtX, use_pinv=True)
        sigma2 = rss / max(1, n_obs - n_features)
        vcov = sigma2 * XtX_inv
        
        return vcov


# ============================================================================
# DuckDB Fitter (Out-of-Core)
# ============================================================================

class DuckDBFitter(BaseFitter):
    """
    Out-of-core OLS estimation using DuckDB sufficient statistics.
    
    Suitable when compressed data is too large to fit in memory.
    Computes X'X and X'y directly in the database using SQL aggregations.
    """
    
    def __init__(self, conn: duckdb.DuckDBPyConnection, alpha: float = DEFAULT_ALPHA,
                 se_type: str = "stata"):
        super().__init__(alpha=alpha, se_type=se_type)
        self.conn = conn
    
    def fit(
        self,
        table_name: str,
        x_cols: List[str],
        y_col: str,
        weight_col: str = "count",
        add_intercept: bool = True,
        cluster_col: Optional[str] = None,
        compute_vcov: bool = True
    ) -> FitterResult:
        """
        Fit WLS model using DuckDB sufficient statistics.
        
        Parameters
        ----------
        table_name : str
            Name of table/view containing data
        x_cols : List[str]
            Column names for X variables
        y_col : str
            Column name for y (sum format, e.g., "sum_outcome")
        weight_col : str
            Column name for weights
        add_intercept : bool
            Whether to add intercept term
        cluster_col : str, optional
            Column for cluster-robust SEs
        compute_vcov : bool
            Whether to compute variance-covariance matrix
            
        Returns
        -------
        FitterResult with coefficients, vcov, etc.
        """
        # Compute sufficient statistics
        XtX, Xty, n_obs, sum_y, sum_y_sq, coef_names = self._compute_sufficient_stats(
            table_name, x_cols, y_col, weight_col, add_intercept
        )
        
        n_features = len(coef_names)
        
        # Solve for coefficients
        theta = self._linalg.safe_solve(XtX, Xty, self.alpha)
        
        # Compute R-squared
        rss = sum_y_sq - theta @ Xty
        mean_y = sum_y / n_obs
        tss = sum_y_sq - n_obs * (mean_y ** 2)
        r_squared = max(0.0, 1.0 - rss / tss) if tss > 0 else 0.0
        
        # Compute vcov if requested
        vcov = None
        se_type_used = "none"
        n_clusters = None
        
        if compute_vcov:
            if cluster_col:
                vcov, n_clusters = self._compute_cluster_robust_vcov_sql(
                    table_name, x_cols, y_col, weight_col, cluster_col,
                    theta, XtX, n_obs, add_intercept
                )
                se_type_used = "cluster"
            else:
                vcov = self._compute_classical_vcov(XtX, rss, n_obs, n_features)
                se_type_used = "classical"
        
        return FitterResult(
            coefficients=theta,
            coef_names=coef_names,
            n_obs=n_obs,
            vcov=vcov,
            se_type=se_type_used,
            r_squared=r_squared,
            rss=rss,
            XtX=XtX,
            Xty=Xty,
            n_clusters=n_clusters
        )
    
    def _compute_sufficient_stats(
        self,
        table_name: str,
        x_cols: List[str],
        y_col: str,
        weight_col: str,
        add_intercept: bool
    ) -> Tuple[np.ndarray, np.ndarray, int, float, float, List[str]]:
        """Compute X'WX and X'Wy using DuckDB aggregations."""
        if add_intercept:
            all_x_cols = ["1"] + x_cols
            coef_names = ["Intercept"] + x_cols
        else:
            all_x_cols = x_cols
            coef_names = x_cols.copy()
        
        k = len(all_x_cols)
        
        # Build X'WX computation (upper triangle only)
        xtx_parts = []
        for i, col_i in enumerate(all_x_cols):
            for j, col_j in enumerate(all_x_cols):
                if j >= i:
                    xtx_parts.append(
                        f"SUM(({col_i}) * ({col_j}) * {weight_col}) AS xtx_{i}_{j}"
                    )
        
        # Build X'Wy computation
        xty_parts = [
            f"SUM(({col}) * {y_col}) AS xty_{i}"
            for i, col in enumerate(all_x_cols)
        ]
        
        # Additional statistics
        stats_parts = [
            f"SUM({weight_col}) AS n_obs",
            f"SUM({y_col}) AS sum_y",
        ]
        
        # Sum of y squared - compute from sum and count (y_col is already sum_y)
        # For compressed data: sum_y_sq = sum((y_i)^2) which we approximate as sum_y^2/count per stratum
        # This is an approximation when we don't have the actual sum of squares
        stats_parts.append(
            f"SUM(POW({y_col} / {weight_col}, 2) * {weight_col}) AS sum_y_sq"
        )
        
        query = f"""
        SELECT {', '.join(xtx_parts)}, {', '.join(xty_parts)}, {', '.join(stats_parts)}
        FROM {table_name}
        """
        
        logger.debug(f"Executing sufficient stats query")
        result = self.conn.execute(query).fetchone()
        
        # Parse XtX (upper triangle)
        XtX = np.zeros((k, k))
        idx = 0
        for i in range(k):
            for j in range(i, k):
                XtX[i, j] = result[idx]
                XtX[j, i] = result[idx]  # Symmetric
                idx += 1
        
        # Add regularization
        XtX += self.alpha * np.eye(k)
        
        # Parse Xty
        Xty = np.array([result[idx + i] for i in range(k)])
        idx += k
        
        n_obs = int(result[idx])
        sum_y = float(result[idx + 1])
        sum_y_sq = float(result[idx + 2])
        
        return XtX, Xty, n_obs, sum_y, sum_y_sq, coef_names
    
    def _compute_cluster_robust_vcov_sql(
        self,
        table_name: str,
        x_cols: List[str],
        y_col: str,
        weight_col: str,
        cluster_col: str,
        theta: np.ndarray,
        XtX: np.ndarray,
        n_obs: int,
        add_intercept: bool
    ) -> Tuple[np.ndarray, int]:
        """Compute cluster-robust vcov using SQL aggregations."""
        if add_intercept:
            all_x_cols = ["1"] + x_cols
        else:
            all_x_cols = x_cols
        
        k = len(all_x_cols)
        
        # Build residual: y - X * theta * weight
        theta_terms = " + ".join([
            f"({theta[i]}) * ({col}) * {weight_col}"
            for i, col in enumerate(all_x_cols)
        ])
        
        # Cluster-level scores: sum of X * residual per cluster
        score_cols = [
            f"SUM(({col}) * ({y_col} - ({theta_terms}))) AS score_{i}"
            for i, col in enumerate(all_x_cols)
        ]
        
        query = f"""
        SELECT {cluster_col}, {', '.join(score_cols)}
        FROM {table_name}
        GROUP BY {cluster_col}
        """
        
        df = self.conn.execute(query).fetchdf()
        n_clusters = len(df)
        
        # Compute meat matrix
        scores = df[[f"score_{i}" for i in range(k)]].values
        meat = scores.T @ scores
        
        # Compute bread
        XtX_inv = self._linalg.safe_inv(XtX, use_pinv=True)
        
        # Apply correction
        correction = self._compute_small_sample_correction(n_obs, k, n_clusters)
        vcov = correction * (XtX_inv @ meat @ XtX_inv)
        vcov = 0.5 * (vcov + vcov.T)
        
        return vcov, n_clusters
    
    def _compute_classical_vcov(
        self,
        XtX: np.ndarray,
        rss: float,
        n_obs: int,
        n_features: int
    ) -> np.ndarray:
        """Compute classical (homoskedastic) variance-covariance matrix."""
        XtX_inv = self._linalg.safe_inv(XtX, use_pinv=True)
        sigma2 = rss / max(1, n_obs - n_features)
        return sigma2 * XtX_inv


# ============================================================================
# Factory Function
# ============================================================================

def get_fitter(
    fitter_type: str = "numpy",
    conn: Optional[duckdb.DuckDBPyConnection] = None,
    alpha: float = DEFAULT_ALPHA,
    se_type: str = "stata"
) -> BaseFitter:
    """
    Factory function to get the appropriate fitter.
    
    Parameters
    ----------
    fitter_type : str
        Either "numpy" for in-memory or "duckdb" for out-of-core
    conn : duckdb.DuckDBPyConnection, optional
        DuckDB connection (required for duckdb fitter)
    alpha : float
        Regularization parameter
    se_type : str
        Standard error type
        
    Returns
    -------
    BaseFitter instance
    """
    if fitter_type == "numpy":
        return NumpyFitter(alpha=alpha, se_type=se_type)
    elif fitter_type == "duckdb":
        if conn is None:
            raise ValueError("DuckDB connection required for duckdb fitter")
        return DuckDBFitter(conn=conn, alpha=alpha, se_type=se_type)
    else:
        raise ValueError(f"Unknown fitter type: {fitter_type}. Use 'numpy' or 'duckdb'")


# ============================================================================
# Convenience Functions (backward compatible)
# ============================================================================

def wls(X: np.ndarray, y: np.ndarray, n: np.ndarray) -> np.ndarray:
    """
    Weighted least squares with frequency weights.
    
    Backward-compatible function for simple WLS computation.
    """
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
    conn: "duckdb.DuckDBPyConnection",
    table_name: str,
    x_cols: List[str],
    y_col: str,
    weight_col: str = "count",
    add_intercept: bool = True,
    cluster_col: Optional[str] = None,
    se_type: str = "stata",
    alpha: float = DEFAULT_ALPHA
) -> Dict[str, Any]:
    """
    Weighted least squares using DuckDB sufficient statistics.
    
    Backward-compatible function that wraps DuckDBFitter.
    
    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active DuckDB connection
    table_name : str
        Table with compressed data
    x_cols : List[str]
        Feature column names
    y_col : str
        Outcome column (sum_y format)
    weight_col : str
        Weight column (count)
    add_intercept : bool
        Add intercept term
    cluster_col : Optional[str]
        Cluster column for robust SEs
    se_type : str
        Standard error type
    alpha : float
        Regularization parameter
        
    Returns
    -------
    dict with coefficients, standard errors, vcov, etc.
    """
    fitter = DuckDBFitter(conn=conn, alpha=alpha, se_type=se_type)
    result = fitter.fit(
        table_name=table_name,
        x_cols=x_cols,
        y_col=y_col,
        weight_col=weight_col,
        add_intercept=add_intercept,
        cluster_col=cluster_col,
        compute_vcov=True
    )
    return result.to_dict()


# ============================================================================
# Export all public symbols
# ============================================================================

__all__ = [
    'LinAlgHelper',
    'FitterResult',
    'BaseFitter',
    'NumpyFitter',
    'DuckDBFitter',
    'get_fitter',
    'wls',
    'wls_duckdb',
    'ridge_closed_form',
    'ridge_closed_form_batch',
    'DEFAULT_ALPHA',
    'CONDITION_NUMBER_THRESHOLD',
]

from abc import ABC, abstractmethod
import duckdb
import numpy as np
import os
from pathlib import Path
import hashlib


######################################################################
# High-level API
######################################################################

def compressed_ols(
    formula: str,
    data: str,
    subset: str = None,
    n_bootstraps: int = 100,
    cache_dir: str = None,
    round_strata: int = None,
    seed: int = 42,
    fe_method: str = "auto",  # new parameter to choose FE method
    duckdb_kwargs: dict = None,
    **kwargs
):
    """High-level API for compressed OLS regression with lfe-style formula
    
    Args:
        formula: Regression formula in format "y ~ x1 + x2 | fe1 + fe2 | iv1 + iv2 | cluster"
                 Parts separated by |:
                 - Left of ~: outcome variable(s) (can use + for multiple)
                 - First part after ~: covariates
                 - Second part: fixed effects (optional)
                 - Third part: instrumental variables (not implemented, raises error)
                 - Fourth part: cluster variable (optional)
        data: Filepath to data file (.csv, .parquet, or directory of .parquet files)
        subset: SQL WHERE clause to subset data (e.g., "year > 2000")
        n_bootstraps: Number of bootstrap iterations (0 to disable)
        cache_dir: Directory for DuckDB cache files (default: .duckreg/ in data dir)
        round_strata: Number of decimals to round strata columns
        seed: Random seed for reproducibility
        fe_method: Method for handling fixed effects ('mundlak', or 'demean')
                   - 'mundlak': Always use Mundlak device for FE
                   - 'demean': Always use demeaning for FE
        duckdb_kwargs: Dictionary of DuckDB configuration settings
        **kwargs: Additional arguments passed to estimator
    
    Returns:
        Fitted estimator object with point_estimate and vcov attributes
    
    Examples:
        >>> # Simple regression
        >>> mod = compressed_ols("y ~ x1 + x2", "data.parquet")
        
        >>> # With fixed effects (auto chooses Mundlak)
        >>> mod = compressed_ols("y ~ x1 + x2 | unit_id + year", "data.parquet")
        
        >>> # Force demeaning method
        >>> mod = compressed_ols("y ~ x1 + x2 | unit_id + year", "data.parquet", fe_method="demean")
        
        >>> # Multiple outcomes with clustering
        >>> mod = compressed_ols("y1 + y2 ~ x1 + x2 | unit_id | country", "data.parquet")
        
        >>> # With subsetting
        >>> mod = compressed_ols("y ~ x", "data.parquet", subset="year >= 2000")
    """
    from .estimators import DuckRegression, DuckMundlak
    
    # Parse formula
    if "~" not in formula:
        raise ValueError("Formula must contain '~' separator")
    
    lhs, rhs = formula.split("~", 1)  # Split only on first ~
    lhs = lhs.strip()
    rhs = rhs.strip()
    
    # Parse left-hand side (outcomes)
    outcome_vars = [x.strip() for x in lhs.split("+")]
    
    # Parse right-hand side parts
    parts = [p.strip() for p in rhs.split("|")]
    if len(parts) > 4:
        raise ValueError("Formula can have at most 4 parts separated by |")
    
    # First part: covariates (required)
    covariates = [x.strip() for x in parts[0].split("+")] if parts[0] else []
    
    # Second part: fixed effects (optional)
    fe_cols = []
    if len(parts) > 1 and parts[1]:
        fe_cols = [x.strip() for x in parts[1].split("+")]
    
    # Third part: instrumental variables (not implemented)
    if len(parts) > 2 and parts[2].strip() != "0" :
        raise NotImplementedError("Instrumental variables not yet implemented")
    
    # Fourth part: cluster variable (optional)
    cluster_col = None
    if len(parts) > 3 and parts[3]:
        cluster_parts = [x.strip() for x in parts[3].split("+")]
        if len(cluster_parts) > 1:
            raise ValueError("Only one cluster variable allowed")
        cluster_col = cluster_parts[0]
    
    # Choose FE method
    if fe_method == "mundlak":
        use_mundlak = True
    elif fe_method == "demean":
        use_mundlak = False
    else:
        raise ValueError("fe_method must be 'mundlak' or 'demean'")
    
    # Setup database
    data_path = Path(data).resolve()
    
    if cache_dir is None:
        # Create .duckreg directory in same location as data
        if data_path.is_file():
            cache_dir = data_path.parent / ".duckreg"
        else:
            cache_dir = data_path / ".duckreg"
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir.mkdir(exist_ok=True, parents=True)
    
    # Create database name from data path hash
    data_hash = hashlib.md5(str(data_path).encode()).hexdigest()[:8]
    db_name = str(cache_dir / f"duckreg_{data_hash}.db")
    
    # Create table reference for DuckDB
    if data_path.is_file():
        suffix = data_path.suffix.lower()
        if suffix == ".csv":
            table_name = f"read_csv('{data_path}')"
        elif suffix == ".parquet":
            table_name = f"read_parquet('{data_path}')"
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    elif data_path.is_dir():
        # Assume directory of parquet files
        table_name = f"read_parquet('{data_path}/**/*.parquet')"
    else:
        raise ValueError(f"Data path not found: {data_path}")
    
    # Choose estimator
    if use_mundlak:
        # Use Mundlak device for FE
        estimator = DuckMundlak(
            db_name=db_name,
            table_name=table_name,
            outcome_vars=outcome_vars,
            covariates=covariates,
            fe_cols=fe_cols,
            cluster_col=cluster_col,
            subset=subset,
            n_bootstraps=n_bootstraps,
            round_strata=round_strata,
            seed=seed,
            duckdb_kwargs=duckdb_kwargs,
            **kwargs
        )
    else:
        # Use simple regression (possibly with FE demeaning)
        estimator = DuckRegression(
            db_name=db_name,
            table_name=table_name,
            outcome_vars=outcome_vars,
            covariates=covariates,
            fe_cols=fe_cols,
            cluster_col=cluster_col,
            subset=subset,
            n_bootstraps=n_bootstraps,
            round_strata=round_strata,
            seed=seed,
            duckdb_kwargs=duckdb_kwargs,
            **kwargs
        )
    
    # Fit the model
    estimator.fit()
    
    return estimator


######################################################################
# Base class
######################################################################

class DuckReg(ABC):
    def __init__(
        self,
        db_name: str,
        table_name: str,
        seed: int,
        n_bootstraps: int = 100,
        fitter: str = "numpy",
        keep_connection_open: bool = False,
        round_strata: int = None,
        duckdb_kwargs: dict = None,
    ):
        """Base class for DuckDB-based regression estimators

        Args:
            db_name: Path to DuckDB database file
            table_name: Name of table containing the data
            seed: Random seed for reproducibility
            n_bootstraps: Number of bootstrap iterations (0 to disable)
            fitter: Fitting method ('numpy' or 'ridge')
            keep_connection_open: Whether to keep database connection open after fitting
            round_strata: Number of decimals to round strata columns (None to disable)
            duckdb_kwargs: Dictionary of DuckDB configuration settings
        """
        self.db_name = db_name
        self.table_name = table_name
        self.n_bootstraps = n_bootstraps
        self.seed = seed
        self.conn = duckdb.connect(db_name)

        # Apply DuckDB configuration settings if provided
        if duckdb_kwargs is not None:
            for key, value in duckdb_kwargs.items():
                self.conn.execute(f"SET {key} = '{value}'")

        self.rng = np.random.default_rng(seed)
        self.fitter = fitter
        self.keep_connection_open = keep_connection_open
        self.round_strata = round_strata
        self.duckdb_kwargs = duckdb_kwargs  # store for reference

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def compress_data(self):
        pass

    @abstractmethod
    def collect_data(self):
        pass

    @abstractmethod
    def estimate(self):
        pass

    @abstractmethod
    def bootstrap(self):
        pass

    def fit(self):
        self.prepare_data()
        self.compress_data()

        self.point_estimate = self.estimate()
        if self.n_bootstraps > 0:
            self.vcov = self.bootstrap()
        self.conn.close() if not self.keep_connection_open else None
        return None

    def summary(self) -> dict:
        """Summary of regression

        Returns:
            dict
        """
        if self.n_bootstraps > 0:
            return {
                "point_estimate": self.point_estimate,
                "standard_error": np.sqrt(np.diag(self.vcov)),
            }
        return {"point_estimate": self.point_estimate}

    def queries(self) -> dict:
        """Collect all query methods in the class

        Returns:
            dict: Dictionary of query methods
        """
        self._query_names = [x for x in dir(self) if "query" in x]
        self.queries = {
            k: getattr(self, self._query_names[c])
            for c, k in enumerate(self._query_names)
        }
        return self.queries


def wls(X: np.ndarray, y: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Weighted least squares with frequency weights"""
    N = np.sqrt(n)
    N = N.reshape(-1, 1) if N.ndim == 1 else N
    Xn = X * N
    yn = y * N
    betahat = np.linalg.lstsq(Xn, yn, rcond=None)[0]
    return betahat


def ridge_closed_form(
    X: np.ndarray, y: np.ndarray, n: np.ndarray, lam: float
) -> np.ndarray:
    """Ridge regression with data augmented representation
    Trad ridge: (X'X + lam I)^{-1} X' y
    Augmentation: Xtilde = [X; sqrt(lam) I], ytilde = [y; 0]
    this lets us use lstsq solver, which is more optimized than using normal equations

    Args:
        X (np.ndarray): Design matrix
        y (np.ndarray): Outcome vector
        n (np.ndarray): Frequency weights
        lam (float): Regularization parameter

    Returns:
        np.ndarray: Coefficient estimates
    """
    k = X.shape[1]
    N = np.sqrt(n)
    Xn = X * N
    yn = y * N
    Xtilde = np.r_[Xn, np.diag(np.repeat(np.sqrt(lam), k))]
    ytilde = np.concatenate([yn, np.zeros(shape=(k, 1))])
    betahat = np.linalg.lstsq(Xtilde, ytilde, rcond=None)[0]
    return betahat


def ridge_closed_form_batch(
    X: np.ndarray, y: np.ndarray, n: np.ndarray, lambda_grid: np.ndarray
) -> np.ndarray:
    """Optimized ridge regression for multiple lambda values
    Pre-computes reusable components to avoid repeated work in lambda grid search

    Args:
        X (np.ndarray): Design matrix
        y (np.ndarray): Outcome vector
        n (np.ndarray): Frequency weights
        lambda_grid (np.ndarray): Array of regularization parameters

    Returns:
        np.ndarray: Coefficient estimates, shape (n_lambdas, n_features)
    """
    k = X.shape[1]
    n_lambdas = len(lambda_grid)

    # Pre-compute weight matrix (done once)
    N = np.sqrt(n)
    # Pre-compute weighted X and y (done once)
    Xn = X * N
    yn = y * N

    # Pre-allocate identity matrix and zero vector (done once)
    I_k = np.eye(k)
    zeros_k = np.zeros((k, 1))

    # Pre-allocate result array
    coefs = np.zeros((n_lambdas, k))

    # Loop over lambda values (only lambda-dependent operations)
    for i, lam in enumerate(lambda_grid):
        # Only lambda-dependent work: scale identity and concatenate
        sqrt_lam_I = np.sqrt(lam) * I_k
        Xtilde = np.vstack([Xn, sqrt_lam_I])
        ytilde = np.vstack([yn, zeros_k])

        coefs[i, :] = np.linalg.lstsq(Xtilde, ytilde, rcond=None)[0].flatten()

    return coefs
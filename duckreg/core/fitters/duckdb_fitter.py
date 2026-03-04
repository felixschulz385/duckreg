"""
Out-of-core OLS estimation using DuckDB sufficient statistics.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any

import duckdb

from ..linalg import safe_solve, safe_inv
from ..vcov import VcovContext, VcovSpec, compute_iid_vcov, compute_hetero_vcov, compute_cluster_vcov
from .. import (
    compute_sufficient_stats_sql,
    compute_residual_aggregates_sql,
)
from ..suffstats import SuffStats, compute_cross_sufficient_stats_sql
from .base import (
    DEFAULT_ALPHA,
    BaseFitter,
    FitterResult,
    _resolve_vcov_spec,
)
from .numpy_fitter import NumpyFitter

logger = logging.getLogger(__name__)


class DuckDBFitter(BaseFitter):
    """Out-of-core OLS estimation using DuckDB sufficient statistics."""

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        alpha: float = DEFAULT_ALPHA,
        se_type: str = "stata",
    ):
        super().__init__(alpha=alpha, se_type=se_type)
        self.conn = conn
        self._last_result: Optional[FitterResult] = None

    # ------------------------------------------------------------------
    # Abstract hook implementations
    # ------------------------------------------------------------------

    def _fetch_suffstats(self, **kwargs) -> SuffStats:
        """Compute SuffStats via DuckDB SQL."""
        table_name = kwargs['table_name']
        x_cols = kwargs.get('x_cols') or kwargs.get('xcols')
        y_col = kwargs.get('y_col') or kwargs.get('ycol')
        weight_col = kwargs.get('weight_col', 'count')
        add_intercept = kwargs.get('add_intercept', True)
        sum_y_sq_col = f"{y_col}_sq"
        return compute_sufficient_stats_sql(
            conn=self.conn, table_name=table_name, x_cols=x_cols,
            y_col=y_col, weight_col=weight_col, add_intercept=add_intercept,
            alpha=self.alpha, sum_y_sq_col=sum_y_sq_col,
        )

    def _fetch_residual_aggregates(
        self, theta, XtX_inv, vcov_spec, n_obs, n_features,
        existing_result, **kwargs,
    ) -> Dict[str, Any]:
        """Delegated to fit_vcov internals."""
        return {}

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

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
        if xcols is not None and x_cols is None:
            x_cols = xcols
        if ycol is not None and y_col is None:
            y_col = ycol
        if weightcol is not None:
            weight_col = weightcol

        sum_y_sq_col = f"{y_col}_sq"
        XtX, Xty, n_obs, sum_y, sum_y_sq, coef_names = compute_sufficient_stats_sql(
            conn=self.conn, table_name=table_name, x_cols=x_cols,
            y_col=y_col, weight_col=weight_col, add_intercept=add_intercept,
            alpha=self.alpha, sum_y_sq_col=sum_y_sq_col,
        )

        theta = (coefficients.flatten() if coefficients is not None
                 else safe_solve(XtX, Xty, self.alpha))

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
            vcov_meta={},
        )
        self._last_result = result
        return result

    # ------------------------------------------------------------------
    # fit_vcov
    # ------------------------------------------------------------------

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
        if xcols is not None and x_cols is None:
            x_cols = xcols
        if ycol is not None and y_col is None:
            y_col = ycol
        if weightcol is not None:
            weight_col = weightcol
        vcov_spec = _resolve_vcov_spec(vcov_spec, vcov_type, ssc_dict, bool(cluster_col))

        result = existing_result or self._last_result

        if result and result.XtX is not None:
            XtX = result.XtX
            Xty = result.Xty
            n_obs = result.n_obs
            coef_names = result.coef_names
        else:
            sum_y_sq_col = f"{y_col}_sq"
            XtX, Xty, n_obs, _, _, coef_names = compute_sufficient_stats_sql(
                conn=self.conn, table_name=table_name, x_cols=x_cols,
                y_col=y_col, weight_col=weight_col, add_intercept=add_intercept,
                alpha=self.alpha, sum_y_sq_col=sum_y_sq_col,
            )

        n_features = XtX.shape[0]

        theta = (coefficients.flatten() if coefficients is not None
                 else result.coefficients.flatten() if result and result.coefficients is not None
                 else safe_solve(XtX, Xty, self.alpha))

        XtX_inv = (result.XtX_inv if result and result.XtX_inv is not None
                   else safe_inv(XtX, use_pinv=True))

        aggregates = {'XtX_inv': XtX_inv, 'theta': theta}

        # IV matrices
        tXZ = tZZinv = tZX = None
        if is_iv and z_cols is not None:
            iv_stats = compute_cross_sufficient_stats_sql(
                conn=self.conn, table_name=table_name,
                x_cols=x_cols, z_cols=z_cols,
                weight_col=weight_col, add_intercept=add_intercept,
            )
            tXZ = iv_stats['tXZ']
            tZZ = iv_stats['tZZ'] + self.alpha * np.eye(iv_stats['tZZ'].shape[0])
            tZZinv = safe_inv(tZZ, use_pinv=True)
            tZX = tXZ.T
            aggregates.update({'tXZ': tXZ, 'tZZinv': tZZinv, 'tZX': tZX})

        # Dispatch
        if cluster_col:
            agg = compute_residual_aggregates_sql(
                theta=theta, conn=self.conn, table_name=table_name,
                x_cols=x_cols, y_col=y_col, weight_col=weight_col,
                cluster_col=cluster_col, add_intercept=add_intercept,
                residual_x_cols=residual_x_cols, compute_cluster_scores=True,
                z_cols=z_cols, is_iv=is_iv,
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
            aggregates.update({
                'cluster_scores': agg['cluster_scores'],
                'n_clusters': agg['n_clusters'],
            })

        elif vcov_spec.vcov_type == 'iid':
            rss = (result.rss if result and result.rss is not None
                   else compute_residual_aggregates_sql(
                       theta=theta, conn=self.conn, table_name=table_name,
                       x_cols=x_cols, y_col=y_col, weight_col=weight_col,
                       add_intercept=add_intercept,
                       residual_x_cols=residual_x_cols, compute_rss=True,
                   )['rss'])
            context = VcovContext(
                N=n_obs, k=n_features, kfe=k_fe, nfe=n_fe,
                kfenested=k_fe_nested, nfefullynested=n_fe_fully_nested,
            )
            vcov, vcov_meta = compute_iid_vcov(
                bread=XtX_inv, rss=rss, context=context,
                ssc_config=vcov_spec.ssc, is_iv=is_iv,
                tXZ=tXZ, tZZinv=tZZinv, tZX=tZX,
            )
            aggregates['rss'] = rss

        else:
            compute_lev = vcov_spec.vcov_detail in ["HC2", "HC3"]
            if compute_lev:
                logger.warning(
                    f"{vcov_spec.vcov_detail} with compressed data: leverages at "
                    f"stratum level. Approximation when strata have multiple observations."
                )
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
                is_iv=is_iv, tXZ=tXZ, tZZinv=tZZinv, tZX=tZX,
            )
            aggregates['meat'] = agg['meat']
            if 'leverages' in agg:
                aggregates['leverages'] = agg['leverages']

        return vcov, vcov_meta, aggregates


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_fitter(
    fitter_type: str = "numpy",
    conn: Optional[duckdb.DuckDBPyConnection] = None,
    **kwargs,
) -> BaseFitter:
    """Get an instance of a fitter class based on type."""
    if fitter_type == "numpy":
        return NumpyFitter(**kwargs)
    elif fitter_type == "duckdb":
        if conn is None:
            raise ValueError("conn must be provided for DuckDB fitter")
        return DuckDBFitter(conn, **kwargs)
    else:
        raise ValueError(f"Unknown fitter type: {fitter_type!r}")


# ---------------------------------------------------------------------------
# Convenience functions (backward compatible)
# ---------------------------------------------------------------------------

def wls(X: np.ndarray, y: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Weighted least squares with frequency weights."""
    N = np.sqrt(n).reshape(-1, 1)
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    return np.linalg.lstsq(X * N, y * N, rcond=None)[0]


def ridge_closed_form(X: np.ndarray, y: np.ndarray, n: np.ndarray, lam: float) -> np.ndarray:
    """Ridge regression in augmented-data form."""
    k = X.shape[1]
    N = np.sqrt(n).reshape(-1, 1)
    Xtilde = np.vstack([X * N, np.sqrt(lam) * np.eye(k)])
    ytilde = np.vstack([y * N, np.zeros((k, 1))])
    return np.linalg.lstsq(Xtilde, ytilde, rcond=None)[0]


def ridge_closed_form_batch(
    X: np.ndarray, y: np.ndarray, n: np.ndarray, lambda_grid: np.ndarray,
) -> np.ndarray:
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
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, Any]:
    """Weighted least squares using DuckDB sufficient statistics."""
    fitter = DuckDBFitter(conn=conn, alpha=alpha, se_type=se_type)
    result = fitter.fit(
        table_name=table_name, x_cols=x_cols, y_col=y_col,
        weight_col=weight_col, add_intercept=add_intercept,
        cluster_col=cluster_col,
    )
    vcov, vcov_meta, _ = fitter.fit_vcov(
        table_name=table_name, x_cols=x_cols, y_col=y_col,
        weight_col=weight_col, add_intercept=add_intercept,
        cluster_col=cluster_col, coefficients=result.coefficients,
        existing_result=result,
    )
    result.vcov = vcov
    result.se_type = vcov_meta.get('vcov_type_detail', se_type)
    result.vcov_meta = vcov_meta
    return result.to_dict()


__all__ = [
    'DuckDBFitter',
    'get_fitter',
    'wls',
    'wls_duckdb',
    'ridge_closed_form',
    'ridge_closed_form_batch',
]

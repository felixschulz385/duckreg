"""
In-memory OLS estimation using numpy.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any

from ..linalg import safe_solve, safe_inv
from ..vcov import VcovSpec
from .. import compute_sufficient_stats_numpy, compute_residual_aggregates_numpy
from ..suffstats import SuffStats
from .base import (
    DEFAULT_ALPHA,
    BaseFitter,
    FitterResult,
    _validate_and_prepare_data,
    _compute_weighted_matrices,
    compute_vcov_dispatch,
    _resolve_vcov_spec,
)

logger = logging.getLogger(__name__)


class NumpyFitter(BaseFitter):
    """In-memory OLS estimation using numpy."""

    def __init__(self, alpha: float = DEFAULT_ALPHA, se_type: str = "stata"):
        super().__init__(alpha=alpha, se_type=se_type)
        self._last_result: Optional[FitterResult] = None

    # ------------------------------------------------------------------
    # Abstract hook implementations
    # ------------------------------------------------------------------

    def _fetch_suffstats(self, **kwargs) -> SuffStats:
        """Compute SuffStats from in-memory arrays."""
        X = kwargs['X']
        y = kwargs['y']
        weights = kwargs['weights']
        coef_names = kwargs.get('coef_names')
        X, y, weights, _, _ = _validate_and_prepare_data(X, y, weights)
        return compute_sufficient_stats_numpy(
            X=X, y=y, weights=weights,
            coef_names=coef_names, alpha=self.alpha,
        )

    def _fetch_residual_aggregates(
        self, theta, XtX_inv, vcov_spec, n_obs, n_features,
        existing_result, **kwargs,
    ) -> Dict[str, Any]:
        """Delegated to compute_vcov_dispatch inside fit_vcov."""
        return {}

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

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
        X, y, weights, n_rows, n_obs = _validate_and_prepare_data(X, y, weights)

        XtX, Xty, n_obs, sum_y, sum_y_sq, coef_names_ss = compute_sufficient_stats_numpy(
            X=X, y=y, weights=weights, coef_names=coef_names, alpha=self.alpha,
        )

        if coef_names is None:
            coef_names = coef_names_ss

        theta = (coefficients.flatten() if coefficients is not None
                 else safe_solve(XtX, Xty.flatten(), self.alpha))

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
            vcov_meta={},
        )
        self._last_result = result
        return result

    # ------------------------------------------------------------------
    # fit_vcov
    # ------------------------------------------------------------------

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
        """Compute variance-covariance matrix."""
        if kfe is not None:
            k_fe = kfe
        if nfe is not None:
            n_fe = nfe
        vcov_spec = _resolve_vcov_spec(vcov_spec, vcov_type, ssc_dict, cluster_ids is not None)

        result = existing_result or self._last_result
        X, y, weights, n_rows, n_obs = _validate_and_prepare_data(X, y, weights)
        n_features = X.shape[1]

        if coefficients is None:
            if result and result.coefficients is not None:
                theta = result.coefficients.flatten()
            else:
                raise ValueError("coefficients must be provided or available in existing_result")
        else:
            theta = coefficients.flatten()

        if result and result.XtX_inv is not None:
            XtX_inv = result.XtX_inv
        else:
            if result and result.XtX is not None:
                XtX = result.XtX
            else:
                XtX, _ = _compute_weighted_matrices(X, y, weights, self.alpha)
            XtX_inv = safe_inv(XtX, use_pinv=True)

        if residual_X is not None:
            residuals = y.flatten() - residual_X @ theta
        else:
            residuals = result.residuals if result and result.residuals is not None else None

        return compute_vcov_dispatch(
            X=X, y=y, weights=weights,
            coefficients=theta, residuals=residuals,
            XtX_inv=XtX_inv, vcov_spec=vcov_spec,
            cluster_ids=cluster_ids,
            k_fe=k_fe, n_fe=n_fe,
            k_fe_nested=k_fe_nested, n_fe_fully_nested=n_fe_fully_nested,
            Z=Z, is_iv=is_iv, alpha=self.alpha,
        )


__all__ = ['NumpyFitter']

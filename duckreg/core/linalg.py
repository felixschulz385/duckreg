"""
Linear algebra operations with numerical stability.

Extracted from LinAlgHelper for reuse across modules.
"""

import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

DEFAULT_ALPHA = 1e-8
CONDITION_NUMBER_THRESHOLD = 1e12

_regularization_warned = False


def safe_solve(
    A: np.ndarray,
    b: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    multiplier: int = 10
) -> np.ndarray:
    """
    Safely solve linear system Ax = b with fallback regularization.
    
    Parameters
    ----------
    A : np.ndarray
        Coefficient matrix (n, n)
    b : np.ndarray
        Right-hand side vector (n,) or (n, m)
    alpha : float
        Base regularization parameter
    multiplier : int
        Multiplier for regularization when needed
        
    Returns
    -------
    np.ndarray
        Solution vector(s)
    """
    global _regularization_warned
    
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        if not _regularization_warned:
            logger.info("Using regularization for numerical stability in solve")
            _regularization_warned = True
        regularized_A = A + alpha * multiplier * np.eye(A.shape[0])
        return np.linalg.solve(regularized_A, b)


def safe_inv(A: np.ndarray, use_pinv: bool = False) -> np.ndarray:
    """
    Safely invert matrix with fallback to pseudo-inverse.
    
    Parameters
    ----------
    A : np.ndarray
        Matrix to invert (n, n)
    use_pinv : bool
        If True, fall back to pseudo-inverse on failure
        
    Returns
    -------
    np.ndarray
        Inverse matrix
        
    Raises
    ------
    np.linalg.LinAlgError
        If inversion fails and use_pinv=False
    """
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        if use_pinv:
            logger.debug("Using pseudo-inverse due to singular matrix")
            return np.linalg.pinv(A)
        raise


def check_condition_number(
    A: np.ndarray,
    threshold: float = CONDITION_NUMBER_THRESHOLD
) -> Tuple[bool, float]:
    """
    Check if matrix is well-conditioned.
    
    Parameters
    ----------
    A : np.ndarray
        Matrix to check (n, n), assumed symmetric
    threshold : float
        Condition number threshold for "well-conditioned"
        
    Returns
    -------
    Tuple[bool, float]
        (is_well_conditioned, condition_number)
    """
    try:
        eigvals = np.linalg.eigvalsh(A)
        cond = eigvals.max() / max(eigvals.min(), 1e-10)
        return cond < threshold, cond
    except np.linalg.LinAlgError:
        return False, float('inf')

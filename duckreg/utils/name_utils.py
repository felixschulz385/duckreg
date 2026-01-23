"""
Centralized coefficient name building utilities.

This module provides functions to build consistent coefficient name lists
across different estimators, eliminating duplication and ensuring consistency.
"""

from typing import List, Tuple, Optional


def build_coef_name_lists(
    formula,
    fe_method: str = "demean",
    include_intercept: bool = True,
    fe_cols: Optional[List[str]] = None,
    is_iv: bool = False,
    endogenous_vars: Optional[List[str]] = None
) -> Tuple[List[str], List[str]]:
    """Build coefficient name lists for display and SQL.
    
    This function centralizes coefficient name building logic that was
    previously duplicated across DuckLinearModel, DuckMundlak, and Duck2SLS.
    
    Args:
        formula: Formula object containing variable specifications
        fe_method: Method for handling fixed effects ('demean' or 'mundlak')
        include_intercept: Whether to include intercept term
        fe_cols: List of fixed effect column names (needed for Mundlak)
        is_iv: Whether this is an IV/2SLS regression
        endogenous_vars: List of endogenous variable display names (for IV)
        
    Returns:
        Tuple of (display_names, sql_names) where:
        - display_names: List for user-facing output
        - sql_names: List for SQL column references
        
    Examples:
        # OLS with demeaning (no intercept, no Mundlak means)
        >>> display, sql = build_coef_name_lists(formula, fe_method='demean', 
        ...                                       include_intercept=False)
        
        # OLS with Mundlak (intercept + covariates + FE averages)
        >>> display, sql = build_coef_name_lists(formula, fe_method='mundlak',
        ...                                       fe_cols=['country', 'year'])
        
        # 2SLS with Mundlak (exogenous + Mundlak means + endogenous)
        >>> display, sql = build_coef_name_lists(formula, fe_method='mundlak',
        ...                                       fe_cols=['country'], is_iv=True,
        ...                                       endogenous_vars=['price'])
    """
    display_names = []
    sql_names = []
    
    # Intercept
    if include_intercept:
        display_names.append('Intercept')
        sql_names.append('1')  # SQL expression for intercept
    
    if not is_iv:
        # Standard OLS case
        simple_covs = [var for var in formula.covariates if not var.is_intercept()]
        
        # Add covariates
        display_names.extend([var.display_name for var in simple_covs])
        sql_names.extend([var.sql_name for var in simple_covs])
        
        # Add Mundlak means if using Mundlak device
        if fe_method == "mundlak" and fe_cols:
            for i in range(len(fe_cols)):
                for var in simple_covs:
                    display_names.append(f"avg_{var.display_name}_fe{i}")
                    sql_names.append(f"avg_{var.sql_name}_fe{i}")
    
    else:
        # 2SLS case
        endogenous_set = set(endogenous_vars or [])
        
        # Exogenous covariates (excluding endogenous)
        for var in formula.covariates:
            if not var.is_intercept() and var.display_name not in endogenous_set:
                display_names.append(var.display_name)
                sql_names.append(var.sql_name)
        
        # Mundlak means for exogenous
        if fe_method == "mundlak" and fe_cols:
            for var in formula.covariates:
                if not var.is_intercept() and var.display_name not in endogenous_set:
                    for i in range(len(fe_cols)):
                        display_names.append(f"avg_{var.display_name}_fe{i}")
                        sql_names.append(f"avg_{var.sql_name}_fe{i}")
            
            # Mundlak means for fitted endogenous
            for var in formula.endogenous:
                for i in range(len(fe_cols)):
                    display_names.append(f"avg_{var.display_name}_fe{i}")
                    sql_names.append(f"avg_fitted_{var.sql_name}_fe{i}")
        
        # Endogenous variables (fitted in second stage)
        for var in formula.endogenous:
            display_names.append(var.display_name)
            sql_names.append(f"fitted_{var.sql_name}")
    
    return display_names, sql_names


def build_first_stage_coef_names(
    formula,
    fe_method: str = "demean",
    fe_cols: Optional[List[str]] = None
) -> List[str]:
    """Build coefficient names for first-stage regression.
    
    First stage uses all exogenous variables (covariates + instruments)
    plus Mundlak means if applicable.
    
    Args:
        formula: Formula object
        fe_method: Method for handling fixed effects
        fe_cols: List of fixed effect column names
        
    Returns:
        List of display names for first stage coefficients
    """
    coef_names = ['Intercept']
    
    # Exogenous covariates (excluding intercept)
    simple_covs = [var for var in formula.covariates if not var.is_intercept()]
    coef_names.extend([var.display_name for var in simple_covs])
    
    # Instruments
    coef_names.extend([var.display_name for var in formula.instruments])
    
    # Mundlak means if applicable
    if fe_method == "mundlak" and fe_cols:
        all_exog = simple_covs + list(formula.instruments)
        for i in range(len(fe_cols)):
            for var in all_exog:
                coef_names.append(f"avg_{var.display_name}_fe{i}")
    
    return coef_names


__all__ = [
    'build_coef_name_lists',
    'build_first_stage_coef_names',
]

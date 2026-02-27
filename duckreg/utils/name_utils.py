"""
Centralized coefficient name building utilities.

This module provides functions to build consistent coefficient name lists
across different estimators, eliminating duplication and ensuring consistency.
"""

from typing import List, Tuple, Optional, Dict, Any


def build_coef_name_lists(
    formula,
    fe_method: str = "demean",
    include_intercept: bool = True,
    fe_cols: Optional[List[str]] = None,
    is_iv: bool = False,
    endogenous_vars: Optional[List[str]] = None,
    fe_metadata: Optional[Dict[str, Dict[str, Any]]] = None
) -> Tuple[List[str], List[str]]:
    """Build coefficient name lists for display and SQL.
    
    This function centralizes coefficient name building logic that was
    previously duplicated across DuckLinearModel, DuckFE, and Duck2SLS.
    
    Args:
        formula: Formula object containing variable specifications
        fe_method: Method for handling fixed effects ('demean' or 'mundlak')
        include_intercept: Whether to include intercept term
        fe_cols: List of fixed effect column names (needed for Mundlak)
        is_iv: Whether this is an IV/2SLS regression
        endogenous_vars: List of endogenous variable display names (for IV)
        fe_metadata: Optional metadata with FE classification and levels info
            Format: {fe_name: {'type': 'fixed'|'asymptotic', 'levels': [...], 'reference_level': ...}}
        
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
        
        # Mundlak with unbalanced panel correction (includes dummy means)
        >>> fe_meta = {'year': {'type': 'fixed', 'levels': [2019, 2020, 2021], 
        ...                     'reference_level': 2019}}
        >>> display, sql = build_coef_name_lists(formula, fe_method='mundlak',
        ...                                       fe_cols=['firm_id', 'year'],
        ...                                       fe_metadata=fe_meta)
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
            # Add standard Mundlak means for asymptotic FEs
            for i, fe_name in enumerate(fe_cols):
                # Check if this FE is asymptotic (default if no metadata)
                is_asymptotic = True
                if fe_metadata and fe_name in fe_metadata:
                    is_asymptotic = fe_metadata[fe_name].get('type') == 'asymptotic'
                
                if is_asymptotic:
                    for var in simple_covs:
                        display_names.append(f"avg_{var.display_name}_fe{i}")
                        sql_names.append(f"avg_{var.sql_name}_fe{i}")
            
            # Add fixed FE dummy columns
            if fe_metadata:
                _add_fixed_fe_dummy_names(display_names, sql_names, fe_cols, fe_metadata)
            
            # Add dummy-mean columns for unbalanced panel correction
            if fe_metadata:
                _add_dummy_mean_names(display_names, sql_names, fe_cols, fe_metadata)
    
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
            for i, fe_name in enumerate(fe_cols):
                # Check if this FE is asymptotic
                is_asymptotic = True
                if fe_metadata and fe_name in fe_metadata:
                    is_asymptotic = fe_metadata[fe_name].get('type') == 'asymptotic'
                
                if is_asymptotic:
                    for var in formula.covariates:
                        if not var.is_intercept() and var.display_name not in endogenous_set:
                            display_names.append(f"avg_{var.display_name}_fe{i}")
                            sql_names.append(f"avg_{var.sql_name}_fe{i}")
                    
                    # Mundlak means for fitted endogenous
                    for var in formula.endogenous:
                        display_names.append(f"avg_{var.display_name}_fe{i}")
                        sql_names.append(f"avg_fitted_{var.sql_name}_fe{i}")
            
            # Add fixed FE dummy columns
            if fe_metadata:
                _add_fixed_fe_dummy_names(display_names, sql_names, fe_cols, fe_metadata)
            
            # Add dummy-mean columns for unbalanced panel correction
            if fe_metadata:
                _add_dummy_mean_names(display_names, sql_names, fe_cols, fe_metadata)
        
        # Endogenous variables (fitted in second stage)
        for var in formula.endogenous:
            display_names.append(var.display_name)
            sql_names.append(f"fitted_{var.sql_name}")
    
    return display_names, sql_names


def _add_fixed_fe_dummy_names(
    display_names: List[str],
    sql_names: List[str],
    fe_cols: List[str],
    fe_metadata: Dict[str, Dict[str, Any]]
):
    """Add fixed FE dummy column names (excluding reference level).
    
    For each fixed FE, adds names for dummy variables representing each level
    except the reference level.
    
    Modifies display_names and sql_names in place.
    """
    # Find fixed FEs
    for i, fe_name in enumerate(fe_cols):
        if fe_name not in fe_metadata:
            continue
        
        meta = fe_metadata[fe_name]
        if meta.get('type') != 'fixed':
            continue
        
        levels = meta.get('levels', [])
        reference_level = meta.get('reference_level')
        fe_sql_name = meta.get('sql_name', fe_name)
        
        # Add names for non-reference levels
        non_ref_levels = [lvl for lvl in levels if lvl != reference_level]
        
        for level in non_ref_levels:
            # Create names like: dummy_year_2020
            dummy_name = f"dummy_{fe_sql_name}_{level}"
            
            display_names.append(dummy_name)
            sql_names.append(dummy_name)


def _add_dummy_mean_names(
    display_names: List[str],
    sql_names: List[str],
    fe_cols: List[str],
    fe_metadata: Dict[str, Dict[str, Any]]
):
    """Add dummy-mean column names for unbalanced panel correction.
    
    For each fixed FE and each asymptotic FE, adds names for
    within-asymptotic-FE means of fixed-FE dummies.
    
    Modifies display_names and sql_names in place.
    """
    # Find fixed and asymptotic FEs
    fixed_fes = []
    asymptotic_fes = []
    
    for i, fe_name in enumerate(fe_cols):
        if fe_name in fe_metadata:
            meta = fe_metadata[fe_name]
            if meta.get('type') == 'fixed':
                fixed_fes.append((i, fe_name, meta))
            elif meta.get('type') == 'asymptotic':
                asymptotic_fes.append((i, fe_name))
    
    # Only add dummy means if we have both fixed and asymptotic FEs
    if not fixed_fes or not asymptotic_fes:
        return
    
    # For each fixed FE and each asymptotic FE, add dummy-mean columns
    for fixed_idx, fixed_fe_name, fixed_meta in fixed_fes:
        levels = fixed_meta.get('levels', [])
        reference_level = fixed_meta.get('reference_level')
        
        # Exclude reference level
        non_ref_levels = [lvl for lvl in levels if lvl != reference_level]
        
        for asymp_idx, asymp_fe_name in asymptotic_fes:
            for level in non_ref_levels:
                # Create names like: avg_year_2020_fe0 (where fe0 is the asymptotic FE)
                display_name = f"avg_{fixed_fe_name}_{level}_fe{asymp_idx}"
                sql_name = f"avg_{fixed_fe_name}_{level}_fe{asymp_idx}"

                display_names.append(display_name)
                sql_names.append(sql_name)


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

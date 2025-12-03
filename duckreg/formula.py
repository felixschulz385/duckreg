import re
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class FormulaParser:
    """Parse and manage lfe-style regression formulas with transformations"""
    
    def __init__(self, formula: str):
        """
        Parse formula in format: "y ~ x1 + x2 | fe1 + fe2 | iv1 + iv2 | cluster"
        
        Supports:
        - Transformations: log(y), log1p(x), log(x + 0.01), exp(z), sqrt(w)
        - Type casting: "y::int ~ x1::float"
        - Lags/Leads: L.x (lag 1), L2.x (lag 2), F.x (lead 1), F3.x (lead 3)
        - Merged FE keys: year*country or year_country for interaction FEs
        - Interactions: a:b (interaction only), a*b (main effects + interaction)
        """
        self.formula = formula
        self.outcome_vars = []
        self.outcome_transforms = {}
        self.outcome_transform_args = {}
        self.outcome_casts = {}
        self.outcome_lags = {}
        
        self.covariates = []
        self.covariate_transforms = {}
        self.covariate_transform_args = {}
        self.covariate_casts = {}
        self.covariate_lags = {}
        self.interactions = []  # List of tuples: (var1, var2, interaction_name)
        
        self.fe_cols = []
        self.fe_casts = {}
        self.fe_merged = {}  # Maps merged FE name -> list of component columns
        
        self.cluster_col = None
        self.cluster_cast = None
        
        self._parse()
    
    def _parse(self):
        """Parse the formula into components"""
        logger.debug(f"Parsing formula: {self.formula}")
        
        if "~" not in self.formula:
            raise ValueError("Formula must contain '~' separator")
        
        lhs, rhs = self.formula.split("~", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        
        # Parse left-hand side (outcomes)
        self.outcome_vars, self.outcome_transforms, self.outcome_transform_args, self.outcome_casts, self.outcome_lags = self._parse_variables(lhs)
        
        # Parse right-hand side parts
        parts = [p.strip() for p in rhs.split("|")]
        if len(parts) > 4:
            raise ValueError("Formula can have at most 4 parts separated by |")
        
        # Parse covariates
        if parts[0]:
            self.covariates, self.covariate_transforms, self.covariate_transform_args, self.covariate_casts, self.covariate_lags = self._parse_variables(parts[0])
        
        # Parse fixed effects with support for merged FE keys
        if len(parts) > 1 and parts[1].strip() != "0":
            self.fe_cols, fe_transforms, fe_transform_args, self.fe_casts, fe_lags = self._parse_variables(parts[1], parse_fe=True)
            if fe_transforms:
                logger.warning("Transformations on fixed effects are ignored")
            if fe_lags:
                logger.warning("Lags/leads on fixed effects are ignored")
        
        # Third part: instrumental variables (not implemented)
        if len(parts) > 2 and parts[2].strip() != "0":
            raise NotImplementedError("Instrumental variables not yet implemented")
        
        # Fourth part: cluster variable
        if len(parts) > 3 and parts[3].strip() != "0":
            cluster_parts, cluster_transforms, cluster_transform_args, cluster_casts, cluster_lags = self._parse_variables(parts[3])
            if len(cluster_parts) > 1:
                raise ValueError("Only one cluster variable allowed")
            self.cluster_col = cluster_parts[0]
            self.cluster_cast = cluster_casts.get(self.cluster_col)
            if cluster_transforms:
                logger.warning("Transformations on cluster variable are ignored")
            if cluster_lags:
                logger.warning("Lags/leads on cluster variable are ignored")
    
    def _split_terms(self, expr: str) -> List[str]:
        """
        Split expression on + signs, but only those outside parentheses.
        
        This ensures we don't split log(x + 0.01) into ['log(x ', ' 0.01)']
        """
        terms = []
        current_term = []
        paren_depth = 0
        
        for char in expr:
            if char == '(':
                paren_depth += 1
                current_term.append(char)
            elif char == ')':
                paren_depth -= 1
                current_term.append(char)
            elif char == '+' and paren_depth == 0:
                # This is a term separator, not part of an expression
                if current_term:
                    terms.append(''.join(current_term).strip())
                    current_term = []
            else:
                current_term.append(char)
        
        # Add the last term
        if current_term:
            terms.append(''.join(current_term).strip())
        
        return terms
    
    def _parse_lag_lead(self, term: str) -> Tuple[Optional[str], Optional[int]]:
        """
        Parse lag/lead prefix from a term using R-style syntax.
        
        Format: L.x or L1.x (lag 1), L2.x (lag 2), F.x or F1.x (lead 1), F3.x (lead 3)
        
        Returns:
            tuple: (base_term_without_lag, lag_value)
                   lag_value: negative for lags, positive for leads, None if no lag/lead
        """
        # Pattern for lag/lead: L[n].var or F[n].var where n is optional or explicit
        # L.x = L1.x (lag 1), L2.x (lag 2), F.x = F1.x (lead 1), etc.
        lag_lead_pattern = r'^(L|F)(\d*)\.(.+)$'
        match = re.match(lag_lead_pattern, term)
        
        if not match:
            return term, None
        
        lag_type = match.group(1)  # 'L' or 'F'
        lag_num = match.group(2)    # Optional number
        base_term = match.group(3)  # Variable name
        
        # Determine lag value
        # If no number specified, default to 1
        # If number is specified (e.g., L2), use that
        if lag_num:
            lag_val = int(lag_num)
        else:
            lag_val = 1
        
        # Negative for lags, positive for leads
        if lag_type == 'L':
            lag_val = -lag_val
        
        return base_term, lag_val
    
    def _parse_variables(self, var_string: str, parse_fe: bool = False) -> Tuple[List[str], Dict[str, str], Dict[str, Dict], Dict[str, str], Dict[str, int]]:
        """
        Parse variable string with optional transformations, type casts, lags/leads, and merged FE keys
        
        Args:
            var_string: String containing variables
            parse_fe: Whether parsing fixed effects (enables merged FE key detection)
        
        Returns:
            tuple: (variables_list, transforms_dict, transform_args_dict, casts_dict, lags_dict)
        """
        variables = []
        transforms = {}
        transform_args = {}
        casts = {}
        lags = {}
        
        # Updated pattern to capture function with complex expressions inside
        transform_pattern = r'^(\w+)\((.*)\)$'
        
        # Split on + but only outside parentheses
        terms = self._split_terms(var_string)
        
        for term in terms:
            term = term.strip()
            if not term:
                continue
            
            # First check for lag/lead prefix (before any other parsing)
            term_no_lag, lag_val = self._parse_lag_lead(term)
            
            # Check for interaction terms (only for covariates, not FE)
            # Must check BEFORE merged FE check since both use *
            if not parse_fe and ('*' in term_no_lag or ':' in term_no_lag):
                # Check if this is a*b (main + interaction) or a:b (interaction only)
                if '*' in term_no_lag and ':' not in term_no_lag:
                    # a*b notation: add both main effects and interaction
                    parts = [p.strip() for p in term_no_lag.split('*')]
                    if len(parts) == 2:
                        # Parse each part to extract variable name and optional cast
                        var1_full = parts[0]
                        var2_full = parts[1]
                        
                        # Extract variable names and casts
                        if '::' in var1_full:
                            var1, cast1 = var1_full.split('::', 1)
                            var1 = var1.strip()
                            cast1 = cast1.strip()
                            casts[var1] = cast1
                        else:
                            var1 = var1_full
                        
                        if '::' in var2_full:
                            var2, cast2 = var2_full.split('::', 1)
                            var2 = var2.strip()
                            cast2 = cast2.strip()
                            casts[var2] = cast2
                        else:
                            var2 = var2_full
                        
                        # Add main effects if not already present
                        if var1 not in variables:
                            variables.append(var1)
                        if var2 not in variables:
                            variables.append(var2)
                        
                        # Add interaction with clean name (no *)
                        interaction_name = f"{var1}_x_{var2}"
                        variables.append(interaction_name)
                        self.interactions.append((var1, var2, interaction_name))
                        
                        cast_info = ""
                        if var1 in casts or var2 in casts:
                            cast_parts = []
                            if var1 in casts:
                                cast_parts.append(f"{var1}::{casts[var1]}")
                            else:
                                cast_parts.append(var1)
                            if var2 in casts:
                                cast_parts.append(f"{var2}::{casts[var2]}")
                            else:
                                cast_parts.append(var2)
                            cast_info = f" ({' * '.join(cast_parts)})"
                        
                        logger.debug(f"  Interaction (a*b): {var1} * {var2} -> {interaction_name}{cast_info}")
                        continue
                        
                elif ':' in term_no_lag:
                    # a:b notation: only add interaction term
                    parts = [p.strip() for p in term_no_lag.split(':')]
                    if len(parts) == 2:
                        # Parse each part to extract variable name and optional cast
                        var1_full = parts[0]
                        var2_full = parts[1]
                        
                        # Extract variable names and casts
                        if '::' in var1_full:
                            var1, cast1 = var1_full.split('::', 1)
                            var1 = var1.strip()
                            cast1 = cast1.strip()
                            casts[var1] = cast1
                        else:
                            var1 = var1_full
                        
                        if '::' in var2_full:
                            var2, cast2 = var2_full.split('::', 1)
                            var2 = var2.strip()
                            cast2 = cast2.strip()
                            casts[var2] = cast2
                        else:
                            var2 = var2_full
                        
                        # Add interaction with clean name (no :)
                        interaction_name = f"{var1}_x_{var2}"
                        variables.append(interaction_name)
                        self.interactions.append((var1, var2, interaction_name))
                        
                        cast_info = ""
                        if var1 in casts or var2 in casts:
                            cast_parts = []
                            if var1 in casts:
                                cast_parts.append(f"{var1}::{casts[var1]}")
                            else:
                                cast_parts.append(var1)
                            if var2 in casts:
                                cast_parts.append(f"{var2}::{casts[var2]}")
                            else:
                                cast_parts.append(var2)
                            cast_info = f" ({' : '.join(cast_parts)})"
                        
                        logger.debug(f"  Interaction (a:b): {var1} : {var2} -> {interaction_name}{cast_info}")
                        continue
            
            # Check for merged FE key (only for FE parsing and only with * separator)
            if parse_fe and '*' in term_no_lag:
                # This is a merged FE key like year*country
                merged_name, component_cols = self._parse_merged_fe(term_no_lag)
                
                # Add the merged FE name to the variables list
                variables.append(merged_name)
                
                # Store the component columns for later use
                self.fe_merged[merged_name] = component_cols
                
                logger.debug(f"  Merged FE: {merged_name} = {' × '.join(component_cols)}")
                continue
            
            # Check for transformation
            transform_match = re.match(transform_pattern, term_no_lag)
            if transform_match:
                func_name = transform_match.group(1)
                inner = transform_match.group(2).strip()
                
                # Parse the inner expression
                var, cast_type, shift = self._parse_inner_expression(inner)
                
                variables.append(var)
                transforms[var] = func_name
                
                # Store transformation arguments
                if shift is not None:
                    transform_args[var] = {'shift': shift}
                
                if cast_type:
                    casts[var] = cast_type
                
                if lag_val is not None:
                    lags[var] = lag_val
                
                lag_str = f" (lag {-lag_val})" if lag_val and lag_val < 0 else f" (lead {lag_val})" if lag_val and lag_val > 0 else ""
                logger.debug(f"  Transform: {func_name}({var}{f' + {shift}' if shift else ''}){lag_str}")
            
            # Check for type cast without transformation
            elif "::" in term_no_lag:
                var, cast_type = term_no_lag.split("::", 1)
                var = var.strip()
                cast_type = cast_type.strip()
                variables.append(var)
                casts[var] = cast_type
                
                if lag_val is not None:
                    lags[var] = lag_val
                
                lag_str = f" (lag {-lag_val})" if lag_val and lag_val < 0 else f" (lead {lag_val})" if lag_val and lag_val > 0 else ""
                logger.debug(f"  Cast: {var} -> {cast_type}{lag_str}")
            
            else:
                variables.append(term_no_lag)
                
                if lag_val is not None:
                    lags[term_no_lag] = lag_val
                    lag_str = f"lag {-lag_val}" if lag_val < 0 else f"lead {lag_val}"
                    logger.debug(f"  Lag/Lead: {term_no_lag} ({lag_str})")
        
        return variables, transforms, transform_args, casts, lags
    
    def _parse_merged_fe(self, term: str) -> Tuple[str, List[str]]:
        """
        Parse a merged fixed effect specification.
        
        Only supports interaction notation with asterisk (*):
        - year*country (creates merged FE 'year_country')
        
        Note: Underscore (_) is NOT supported as a separator since it's commonly
        used in column names. Use asterisk (*) for FE interactions.
        
        Args:
            term: FE term containing * separator
        
        Returns:
            tuple: (merged_name, list of component columns)
        """
        if '*' not in term:
            raise ValueError(f"Merged FE must use * separator: {term}")
        
        # Split by asterisk
        component_cols = [col.strip() for col in term.split('*')]
        
        # Validate that all components are simple column names (no transformations)
        for col in component_cols:
            if '(' in col or ')' in col or '::' in col:
                raise ValueError(f"Merged FE components cannot have transformations or casts: {col}")
        
        # Create merged name by joining with underscore
        merged_name = '_'.join(component_cols)
        
        return merged_name, component_cols
    
    def _parse_inner_expression(self, expr: str) -> Tuple[str, Optional[str], Optional[float]]:
        """
        Parse inner expression of transformation, e.g., "var + 0.01" or "var::float + 0.01"
        
        Returns:
            tuple: (variable_name, cast_type, shift_constant)
        """
        var = expr
        cast_type = None
        shift = None
        
        # Check for shift (addition or subtraction)
        # Pattern: variable [::type] [+/- number]
        shift_pattern = r'^(.+?)\s*([+\-])\s*([0-9.]+)$'
        shift_match = re.match(shift_pattern, expr)
        
        if shift_match:
            var = shift_match.group(1).strip()
            operator = shift_match.group(2)
            value = float(shift_match.group(3))
            shift = value if operator == '+' else -value
        
        # Check for type cast in the variable part
        if "::" in var:
            var, cast_type = var.split("::", 1)
            var = var.strip()
            cast_type = cast_type.strip()
        
        return var, cast_type, shift

    def get_all_raw_columns(self) -> List[str]:
        """Get all raw column names needed from data (before transformations)"""
        cols = []
        cols.extend(self.outcome_vars)
        cols.extend(self.covariates)
        
        # For FE columns, include both simple and merged FE components
        for fe_col in self.fe_cols:
            if fe_col in self.fe_merged:
                # Add component columns for merged FEs
                cols.extend(self.fe_merged[fe_col])
            else:
                # Add simple FE column
                cols.append(fe_col)
        
        if self.cluster_col:
            cols.append(self.cluster_col)
        return cols
    
    def get_all_casts(self) -> Dict[str, str]:
        """Get all type casts as a single dictionary"""
        all_casts = {}
        all_casts.update(self.outcome_casts)
        all_casts.update(self.covariate_casts)
        all_casts.update(self.fe_casts)
        if self.cluster_cast:
            all_casts[self.cluster_col] = self.cluster_cast
        return all_casts
    
    def get_sql_transform(self, var: str, var_type: str = 'covariate', 
                         time_col: str = 'year', unit_col: str = None) -> str:
        """
        Get SQL expression for transforming a variable, including lags/leads
        
        Args:
            var: Variable name
            var_type: Type of variable ('covariate' or 'outcome')
            time_col: Name of time column for lag/lead operations
            unit_col: Name of unit column for PARTITION BY (required for panel data)
        
        Returns:
            SQL expression (e.g., "LOG(x + 0.01)" or "LAG(x, 1) OVER (PARTITION BY id ORDER BY year)")
        """
        if var_type == 'covariate':
            transforms = self.covariate_transforms
            transform_args = self.covariate_transform_args
            lags = self.covariate_lags
        elif var_type == 'outcome':
            transforms = self.outcome_transforms
            transform_args = self.outcome_transform_args
            lags = self.outcome_lags
        else:
            return var
        
        # Start with base variable
        base_expr = var
        
        # Apply transformation if exists
        if var in transforms:
            transform = transforms[var]
            args = transform_args.get(var, {})
            shift = args.get('shift', 0)
            
            # Build expression with transformation
            if shift != 0:
                base_expr = f"({var} + {shift})"
            
            if transform == 'log':
                base_expr = f"LN({base_expr})"
            elif transform == 'log1p':
                if shift != 0:
                    base_expr = f"LN({base_expr} + 1)"
                else:
                    base_expr = f"LN({var} + 1)"
            elif transform == 'exp':
                base_expr = f"EXP({base_expr})"
            elif transform == 'sqrt':
                base_expr = f"SQRT({base_expr})"
            else:
                logger.warning(f"Unknown transformation '{transform}' for {var}")
        
        # Apply lag/lead if exists
        if var in lags:
            lag_val = lags[var]
            
            # Build OVER clause with PARTITION BY if unit_col provided
            if unit_col:
                over_clause = f"OVER (PARTITION BY {unit_col} ORDER BY {time_col})"
            else:
                over_clause = f"OVER (ORDER BY {time_col})"
            
            if lag_val < 0:
                # Lag: use LAG function
                base_expr = f"LAG({base_expr}, {-lag_val}) {over_clause}"
            elif lag_val > 0:
                # Lead: use LEAD function
                base_expr = f"LEAD({base_expr}, {lag_val}) {over_clause}"
        
        return base_expr
    
    def get_sql_interaction(self, var1: str, var2: str, interaction_name: str) -> str:
        """
        Get SQL expression for creating an interaction term
        
        Args:
            var1: First variable name
            var2: Second variable name
            interaction_name: Name for the interaction column
        
        Returns:
            SQL expression (e.g., "x1 * x2 AS x1_x2")
        """
        return f"({var1} * {var2}) AS {interaction_name}"
    
    def apply_transformations(self, df):
        """
        Apply transformations to a DataFrame (after compression)
        
        This is now mainly for the demean estimator. For Mundlak,
        transformations are applied at SQL level before compression.
        
        Args:
            df: pandas DataFrame with raw columns
        
        Returns:
            DataFrame with transformed columns added
        """
        import numpy as np
        
        # Apply outcome transformations
        for var, transform in self.outcome_transforms.items():
            col_name = f'mean_{var}'
            args = self.outcome_transform_args.get(var, {})
            shift = args.get('shift', 0)
            
            if transform == 'log':
                if shift != 0:
                    df[col_name] = np.log(df[col_name] + shift)
                else:
                    df[col_name] = np.log(df[col_name])
            elif transform == 'log1p':
                if shift != 0:
                    df[col_name] = np.log1p(df[col_name] + shift)
                else:
                    df[col_name] = np.log1p(df[col_name])
            elif transform == 'exp':
                if shift != 0:
                    df[col_name] = np.exp(df[col_name] + shift)
                else:
                    df[col_name] = np.exp(df[col_name])
            elif transform == 'sqrt':
                if shift != 0:
                    df[col_name] = np.sqrt(df[col_name] + shift)
                else:
                    df[col_name] = np.sqrt(df[col_name])
            else:
                logger.warning(f"Unknown transformation '{transform}' for {var}")
        
        # Apply covariate transformations
        for var, transform in self.covariate_transforms.items():
            args = self.covariate_transform_args.get(var, {})
            shift = args.get('shift', 0)
            
            if transform == 'log':
                if shift != 0:
                    df[var] = np.log(df[var] + shift)
                else:
                    df[var] = np.log(df[var])
            elif transform == 'log1p':
                if shift != 0:
                    df[var] = np.log1p(df[var] + shift)
                else:
                    df[var] = np.log1p(df[var])
            elif transform == 'exp':
                if shift != 0:
                    df[var] = np.exp(df[var] + shift)
                else:
                    df[var] = np.exp(df[var])
            elif transform == 'sqrt':
                if shift != 0:
                    df[var] = np.sqrt(df[var] + shift)
                else:
                    df[var] = np.sqrt(df[var])
            else:
                logger.warning(f"Unknown transformation '{transform}' for {var}")
        
        return df
    
    def __repr__(self):
        return f"FormulaParser('{self.formula}')"

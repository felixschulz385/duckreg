"""
Formula parsing with clean abstractions following software engineering principles.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


class VariableRole(Enum):
    """Role of a variable in the regression formula"""
    OUTCOME = "outcome"
    COVARIATE = "covariate"
    FIXED_EFFECT = "fixed_effect"
    CLUSTER = "cluster"


class TransformType(Enum):
    """Supported transformation functions"""
    NONE = "none"
    LOG = "log"
    LOG1P = "log1p"
    EXP = "exp"
    SQRT = "sqrt"
    
    def to_sql(self, expr: str) -> str:
        """Convert transformation to SQL expression"""
        sql_funcs = {
            TransformType.LOG: f"LN({expr})",
            TransformType.LOG1P: f"LN({expr} + 1)",
            TransformType.EXP: f"EXP({expr})",
            TransformType.SQRT: f"SQRT({expr})",
        }
        return sql_funcs.get(self, expr)


def _cast_if_boolean(expr: str, col_name: str, boolean_cols: Set[str]) -> str:
    """Helper to cast expression if column is boolean"""
    if boolean_cols and col_name in boolean_cols:
        return f"CAST({expr} AS SMALLINT)"
    return expr


@dataclass(frozen=True)
class Variable:
    """Immutable representation of a parsed variable"""
    name: str
    role: VariableRole
    transform: TransformType = TransformType.NONE
    transform_shift: float = 0.0
    lag: Optional[int] = None
    display_name: Optional[str] = None
    
    def __post_init__(self):
        if self.display_name is None:
            object.__setattr__(self, 'display_name', self.name)
    
    def get_sql_expression(self, unit_col: Optional[str] = None, 
                          time_col: str = 'year') -> str:
        """Generate SQL expression for this variable"""
        expr = self.name
        
        if self.transform != TransformType.NONE:
            if self.transform_shift != 0:
                expr = f"({expr} + {self.transform_shift})"
            expr = self.transform.to_sql(expr)
        
        if self.lag is not None:
            partition = f"PARTITION BY {unit_col} " if unit_col else ""
            over_clause = f"OVER ({partition}ORDER BY {time_col})"
            func = "LAG" if self.lag < 0 else "LEAD"
            offset = abs(self.lag)
            expr = f"{func}({expr}, {offset}) {over_clause}"
        
        return expr
    
    def get_select_sql(self, unit_col: Optional[str] = None, 
                       time_col: str = 'year',
                       boolean_cols: Set[str] = None) -> str:
        """Generate SELECT clause fragment: expression AS alias"""
        expr = self.get_sql_expression(unit_col, time_col)
        expr = _cast_if_boolean(expr, self.name, boolean_cols or set())
        return f"{expr} AS {self.name}"


@dataclass(frozen=True)
class Interaction:
    """Immutable representation of an interaction term"""
    var1: Variable
    var2: Variable
    name: str
    include_main_effects: bool = True
    
    def get_sql_expression(self, unit_col: Optional[str] = None, 
                           time_col: str = 'year',
                           boolean_cols: Set[str] = None) -> str:
        """Generate SQL expression for interaction (without alias)"""
        boolean_cols = boolean_cols or set()
        
        expr1 = self.var1.get_sql_expression(unit_col, time_col)
        expr2 = self.var2.get_sql_expression(unit_col, time_col)
        
        expr1 = _cast_if_boolean(expr1, self.var1.name, boolean_cols)
        expr2 = _cast_if_boolean(expr2, self.var2.name, boolean_cols)
        
        return f"({expr1} * {expr2})"
    
    def get_select_sql(self, unit_col: Optional[str] = None,
                       time_col: str = 'year',
                       boolean_cols: Set[str] = None) -> str:
        """Generate SELECT clause fragment: expression AS alias"""
        expr = self.get_sql_expression(unit_col, time_col, boolean_cols)
        return f"{expr} AS {self.name}"


@dataclass(frozen=True)
class MergedFixedEffect:
    """Immutable representation of a merged fixed effect"""
    name: str
    components: Tuple[Variable, ...]
    use_numeric_merge: bool = False
    
    _NUMERIC_MULTIPLIER = 1_000_000_000  # Constant for numeric merge
    
    def get_sql_expression(self, boolean_cols: Set[str] = None) -> str:
        """Generate SQL expression for merged FE (without alias)"""
        if self.use_numeric_merge and len(self.components) == 2:
            comp_exprs = [f"CAST({c.get_sql_expression()} AS BIGINT)" for c in self.components]
            return f"({comp_exprs[0]} * {self._NUMERIC_MULTIPLIER} + {comp_exprs[1]})"
        
        comp_exprs = [f"CAST({c.get_sql_expression()} AS VARCHAR)" for c in self.components]
        return "(" + " || '_' || ".join(comp_exprs) + ")"
    
    def get_select_sql(self, boolean_cols: Set[str] = None) -> str:
        """Generate SELECT clause fragment: expression AS alias"""
        return f"{self.get_sql_expression(boolean_cols)} AS {self.name}"
    
    def get_source_column_names(self) -> List[str]:
        """Get the actual source column names (for NULL checks)"""
        return [comp.name for comp in self.components]


@dataclass(frozen=True)
class Formula:
    """Immutable representation of a parsed regression formula"""
    outcomes: Tuple[Variable, ...]
    covariates: Tuple[Variable, ...]
    interactions: Tuple[Interaction, ...]
    fixed_effects: Tuple[Variable, ...]
    merged_fes: Tuple[MergedFixedEffect, ...]
    cluster: Optional[Variable]
    raw_formula: str
    
    # -------------------------------------------------------------------------
    # Generic lookup helper
    # -------------------------------------------------------------------------
    
    def _find_in_tuple(self, items: tuple, name: str):
        """Generic lookup by name in a tuple of objects with .name attribute"""
        return next((item for item in items if item.name == name), None)
    
    # -------------------------------------------------------------------------
    # Basic name getters
    # -------------------------------------------------------------------------
    
    def get_outcome_names(self) -> List[str]:
        return [var.name for var in self.outcomes]
    
    def get_covariate_names(self) -> List[str]:
        """Get all covariate names including interactions"""
        return [var.name for var in self.covariates] + [i.name for i in self.interactions]
    
    def get_simple_covariate_names(self) -> List[str]:
        """Get only simple covariate names (no interactions)"""
        return [var.name for var in self.covariates]
    
    def get_fe_names(self) -> List[str]:
        """Get all FE names including merged FEs"""
        return [var.name for var in self.fixed_effects] + [mfe.name for mfe in self.merged_fes]
    
    def get_simple_fe_names(self) -> List[str]:
        """Get only simple FE names (no merged FEs)"""
        return [var.name for var in self.fixed_effects]
    
    # -------------------------------------------------------------------------
    # Lookup methods (using generic helper)
    # -------------------------------------------------------------------------
    
    def get_interaction_by_name(self, name: str) -> Optional[Interaction]:
        return self._find_in_tuple(self.interactions, name)
    
    def get_merged_fe_by_name(self, name: str) -> Optional[MergedFixedEffect]:
        return self._find_in_tuple(self.merged_fes, name)
    
    def get_covariate_by_name(self, name: str) -> Optional[Variable]:
        return self._find_in_tuple(self.covariates, name)
    
    def get_fe_by_name(self, name: str) -> Optional[Variable]:
        return self._find_in_tuple(self.fixed_effects, name)
    
    # -------------------------------------------------------------------------
    # Source column methods (for NULL checks in WHERE clause)
    # -------------------------------------------------------------------------
    
    def get_source_columns_for_null_check(self) -> List[str]:
        """Get all source column names needed for NULL checking."""
        cols = set()
        
        # Add names from simple variables
        for var_list in (self.outcomes, self.covariates, self.fixed_effects):
            cols.update(var.name for var in var_list)
        
        # Interaction components
        for interaction in self.interactions:
            cols.add(interaction.var1.name)
            cols.add(interaction.var2.name)
        
        # Merged FE components
        for mfe in self.merged_fes:
            cols.update(comp.name for comp in mfe.components)
        
        if self.cluster:
            cols.add(self.cluster.name)
        
        return list(cols)
    
    # -------------------------------------------------------------------------
    # SQL generation methods
    # -------------------------------------------------------------------------
    
    def _join_select_parts(self, parts: List[str]) -> str:
        """Join non-empty SELECT parts with comma"""
        return ", ".join(p for p in parts if p)
    
    def get_outcomes_select_sql(self, unit_col: Optional[str] = None,
                                time_col: str = 'year',
                                boolean_cols: Set[str] = None) -> str:
        """Generate SELECT clause for outcome variables"""
        return self._join_select_parts([
            var.get_select_sql(unit_col, time_col, boolean_cols) 
            for var in self.outcomes
        ])
    
    def get_covariates_select_sql(self, unit_col: Optional[str] = None,
                                  time_col: str = 'year',
                                  boolean_cols: Set[str] = None,
                                  include_interactions: bool = True) -> str:
        """Generate SELECT clause for covariates (optionally including interactions)"""
        boolean_cols = boolean_cols or set()
        parts = [var.get_select_sql(unit_col, time_col, boolean_cols) for var in self.covariates]
        
        if include_interactions:
            parts.extend(i.get_select_sql(unit_col, time_col, boolean_cols) for i in self.interactions)
        
        return self._join_select_parts(parts)
    
    def get_fe_select_sql(self, boolean_cols: Set[str] = None) -> str:
        """Generate SELECT clause for fixed effects (including merged FEs)"""
        boolean_cols = boolean_cols or set()
        parts = []
        
        for var in self.fixed_effects:
            expr = _cast_if_boolean(var.get_sql_expression(), var.name, boolean_cols)
            parts.append(f"{expr} AS {var.name}")
        
        parts.extend(mfe.get_select_sql(boolean_cols) for mfe in self.merged_fes)
        return self._join_select_parts(parts)
    
    def get_cluster_select_sql(self, boolean_cols: Set[str] = None,
                               alias: str = "__cluster__",
                               fallback_fe: Optional[str] = None) -> str:
        """Generate SELECT clause for cluster variable."""
        boolean_cols = boolean_cols or set()
        
        # Determine which column to use
        if self.cluster:
            col = self.cluster.name
        elif fallback_fe:
            col = fallback_fe
        elif self.get_fe_names():
            col = self.get_fe_names()[0]
        else:
            col = None
        
        if col is None:
            return ""
        
        expr = _cast_if_boolean(col, col, boolean_cols)
        return f"{expr} AS {alias}"
    
    def get_where_clause_sql(self, user_subset: Optional[str] = None) -> str:
        """Generate complete WHERE clause with NULL check and optional user subset"""
        cols = self.get_source_columns_for_null_check()
        null_check = f"COLUMNS([{', '.join(repr(c) for c in cols)}]) IS NOT NULL" if cols else "1=1"
        
        if user_subset:
            return f"WHERE {null_check} AND ({user_subset})"
        return f"WHERE {null_check}"
    
    # -------------------------------------------------------------------------
    # Aggregation SQL methods
    # -------------------------------------------------------------------------
    
    def get_outcome_agg_sql(self, include_sum_sq: bool = True) -> str:
        """Generate aggregation expressions for outcomes (SUM, SUM of squares)"""
        parts = []
        for var in self.outcomes:
            expr = var.get_sql_expression()
            parts.append(f"SUM({expr}) AS sum_{var.name}")
            if include_sum_sq:
                parts.append(f"SUM(POW({expr}, 2)) AS sum_{var.name}_sq")
        return self._join_select_parts(parts)
    
    def get_covariate_expression(self, cov_name: str,
                                 unit_col: Optional[str] = None,
                                 time_col: str = 'year',
                                 boolean_cols: Set[str] = None) -> str:
        """Get SQL expression for a covariate by name (handles interactions)"""
        boolean_cols = boolean_cols or set()
        
        interaction = self.get_interaction_by_name(cov_name)
        if interaction:
            return interaction.get_sql_expression(unit_col, time_col, boolean_cols)
        
        cov_var = self.get_covariate_by_name(cov_name)
        if cov_var:
            expr = cov_var.get_sql_expression(unit_col, time_col)
            return _cast_if_boolean(expr, cov_var.name, boolean_cols)
        
        return cov_name
    
    def get_fe_expression(self, fe_name: str, boolean_cols: Set[str] = None) -> str:
        """Get SQL expression for an FE by name (handles merged FEs)"""
        boolean_cols = boolean_cols or set()
        
        mfe = self.get_merged_fe_by_name(fe_name)
        if mfe:
            return mfe.get_sql_expression(boolean_cols)
        
        fe_var = self.get_fe_by_name(fe_name)
        if fe_var:
            expr = fe_var.get_sql_expression()
            return _cast_if_boolean(expr, fe_var.name, boolean_cols)
        
        return fe_name


class FormulaParser:
    """Parser for lfe-style regression formulas"""
    
    _TRANSFORM_PATTERN = re.compile(r'^(\w+)\((.*)\)$')
    _LAG_LEAD_PATTERN = re.compile(r'^(L|F)(\d*)\.(.+)$')
    _SHIFT_PATTERN = re.compile(r'^(.+?)\s*([+\-])\s*([0-9.]+)$')
    
    def parse(self, formula: str) -> Formula:
        """Parse a formula string into a Formula object"""
        logger.debug(f"Parsing formula: {formula}")
        
        if "~" not in formula:
            raise ValueError("Formula must contain '~' separator")
        
        lhs, rhs = formula.split("~", 1)
        outcomes = self._parse_variable_list(lhs.strip(), VariableRole.OUTCOME)
        
        parts = [p.strip() for p in rhs.split("|")]
        if len(parts) > 4:
            raise ValueError("Formula can have at most 4 parts separated by |")
        
        covariates, interactions = self._parse_covariates_with_interactions(parts[0]) if parts[0] else ([], [])
        fixed_effects, merged_fes = self._parse_fixed_effects(parts[1]) if len(parts) > 1 and parts[1].strip() != "0" else ([], [])
        
        if len(parts) > 2 and parts[2].strip() not in ("", "0"):
            raise NotImplementedError("Instrumental variables not yet implemented")
        
        cluster = None
        if len(parts) > 3 and parts[3].strip() not in ("", "0"):
            cluster_vars = self._parse_variable_list(parts[3], VariableRole.CLUSTER)
            if len(cluster_vars) > 1:
                raise ValueError("Only one cluster variable allowed")
            cluster = cluster_vars[0] if cluster_vars else None
        
        return Formula(
            outcomes=tuple(outcomes),
            covariates=tuple(covariates),
            interactions=tuple(interactions),
            fixed_effects=tuple(fixed_effects),
            merged_fes=tuple(merged_fes),
            cluster=cluster,
            raw_formula=formula
        )
    
    def _split_terms(self, expr: str) -> List[str]:
        """Split expression on + but only outside parentheses"""
        terms, current, depth = [], [], 0
        
        for char in expr:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            
            if char == '+' and depth == 0:
                if current:
                    terms.append(''.join(current).strip())
                    current = []
            else:
                current.append(char)
        
        if current:
            terms.append(''.join(current).strip())
        return terms
    
    def _parse_variable_list(self, var_string: str, role: VariableRole) -> List[Variable]:
        """Parse a list of simple variables"""
        return [self._parse_single_variable(term.strip(), role) 
                for term in self._split_terms(var_string) if term.strip()]
    
    def _parse_covariates_with_interactions(self, cov_string: str) -> Tuple[List[Variable], List[Interaction]]:
        """Parse covariates including interaction terms"""
        covariates, interactions = [], []
        seen_names = set()
        
        for term in self._split_terms(cov_string):
            term = term.strip()
            if not term:
                continue
            
            interaction_sep = self._find_interaction_separator(term)
            
            if interaction_sep:
                sep, include_main = interaction_sep
                parts = [p.strip() for p in term.split(sep)]
                if len(parts) == 2:
                    var1 = self._parse_single_variable(parts[0], VariableRole.COVARIATE)
                    var2 = self._parse_single_variable(parts[1], VariableRole.COVARIATE)
                    
                    if include_main:
                        for var in (var1, var2):
                            if var.name not in seen_names:
                                covariates.append(var)
                                seen_names.add(var.name)
                    
                    interactions.append(Interaction(var1, var2, f"{var1.name}_x_{var2.name}", include_main))
            else:
                var = self._parse_single_variable(term, VariableRole.COVARIATE)
                if var.name not in seen_names:
                    covariates.append(var)
                    seen_names.add(var.name)
        
        return covariates, interactions
    
    def _find_interaction_separator(self, term: str) -> Optional[Tuple[str, bool]]:
        """Find interaction separator (* or :) and return (separator, include_main_effects)"""
        if '*' in term:
            # Check it's not inside a cast (::)
            if '::' not in term or term.index('*') < term.index('::'):
                return ('*', True)
        
        # Check for single colon (not double colon for cast)
        for i, char in enumerate(term):
            if char == ':' and (i + 1 >= len(term) or term[i + 1] != ':'):
                return (':', False)
        
        return None
    
    def _parse_fixed_effects(self, fe_string: str) -> Tuple[List[Variable], List[MergedFixedEffect]]:
        """Parse fixed effects including merged FEs"""
        fixed_effects, merged_fes = [], []
        
        for term in self._split_terms(fe_string):
            term = term.strip()
            if not term:
                continue
            
            if '*' in term:
                parts = [p.strip() for p in term.split('*')]
                components = tuple(self._parse_single_variable(p, VariableRole.FIXED_EFFECT) for p in parts)
                merged_name = '_'.join(c.name for c in components)
                merged_fes.append(MergedFixedEffect(merged_name, components, len(components) == 2))
            else:
                fixed_effects.append(self._parse_single_variable(term, VariableRole.FIXED_EFFECT))
        
        return fixed_effects, merged_fes
    
    def _parse_single_variable(self, term: str, role: VariableRole) -> Variable:
        """Parse a single variable term"""
        term_no_lag, lag_val = self._parse_lag_lead(term)
        
        match = self._TRANSFORM_PATTERN.match(term_no_lag)
        if match:
            func_name = match.group(1).upper()
            var_name, shift = self._parse_inner_expression(match.group(2).strip())
            transform = TransformType[func_name] if func_name in TransformType.__members__ else TransformType.NONE
            return Variable(name=var_name, role=role, transform=transform, transform_shift=shift, lag=lag_val)
        
        return Variable(name=term_no_lag, role=role, lag=lag_val)
    
    def _parse_lag_lead(self, term: str) -> Tuple[str, Optional[int]]:
        """Parse lag/lead prefix"""
        match = self._LAG_LEAD_PATTERN.match(term)
        if not match:
            return term, None
        
        lag_num = int(match.group(2)) if match.group(2) else 1
        return match.group(3), -lag_num if match.group(1) == 'L' else lag_num
    
    def _parse_inner_expression(self, expr: str) -> Tuple[str, float]:
        """Parse inner expression of transformation"""
        match = self._SHIFT_PATTERN.match(expr)
        if match:
            var = match.group(1).strip()
            value = float(match.group(3))
            return var, value if match.group(2) == '+' else -value
        return expr, 0.0
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
    ENDOGENOUS = "endogenous"
    INSTRUMENT = "instrument"


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


# ============================================================================
# Centralized Quoting Logic
# ============================================================================

# SQL reserved words that require quoting
_SQL_RESERVED_WORDS = frozenset({
    'SELECT', 'FROM', 'WHERE', 'GROUP', 'ORDER', 'BY', 'AS', 'AND', 'OR', 
    'NOT', 'NULL', 'TRUE', 'FALSE', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER',
    'ON', 'IN', 'IS', 'LIKE', 'BETWEEN', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END',
    'HAVING', 'LIMIT', 'OFFSET', 'UNION', 'ALL', 'DISTINCT', 'CREATE', 'TABLE',
    'INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'INDEX', 'VIEW', 'WITH',
    'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'CAST', 'AS'
})


def needs_quoting(name: str) -> bool:
    """
    Check if a column/identifier name needs to be quoted in SQL.
    
    Returns True if the name:
    - Contains special characters (not alphanumeric or underscore)
    - Starts with a digit
    - Is a SQL reserved word
    - Contains spaces, parentheses, operators, etc.
    
    Args:
        name: The identifier name to check
        
    Returns:
        True if quoting is needed, False otherwise
    """
    if not name:
        return True
    
    # Check for numeric names or names starting with digit
    if name.isdigit() or name[0].isdigit():
        return True
    
    # Check if it's a valid Python identifier (alphanumeric + underscore, not starting with digit)
    # This catches most special characters
    if not name.replace('_', '').replace('.', 'x').isalnum():
        return True
    
    # More thorough check for special characters
    for char in name:
        if not (char.isalnum() or char == '_'):
            return True
    
    # Check for SQL reserved words (case-insensitive)
    if name.upper() in _SQL_RESERVED_WORDS:
        return True
    
    return False


def quote_identifier(name: str) -> str:
    """
    Quote an identifier if needed for SQL safety.
    
    Args:
        name: The identifier name to potentially quote
        
    Returns:
        The name, quoted with double quotes if necessary
    """
    # If already quoted, return as-is
    if name.startswith('"') and name.endswith('"'):
        return name
    
    if needs_quoting(name):
        # Escape any existing double quotes by doubling them
        escaped = name.replace('"', '""')
        return f'"{escaped}"'
    return name


def quote_identifier_always(name: str) -> str:
    """
    Always quote an identifier (for cases where we want consistent quoting).
    
    Args:
        name: The identifier name to quote
        
    Returns:
        The name quoted with double quotes
    """
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def cast_if_boolean(expr: str, col_name: str, boolean_cols: Set[str]) -> str:
    """
    Cast expression to SMALLINT if the column is boolean.
    
    Args:
        expr: The SQL expression
        col_name: The column name to check
        boolean_cols: Set of known boolean column names
        
    Returns:
        The expression, wrapped in CAST if boolean
    """
    if boolean_cols and col_name in boolean_cols:
        return f"CAST({expr} AS SMALLINT)"
    return expr


# Keep backward-compatible aliases with underscore prefix
_needs_quoting = needs_quoting
_quote_identifier = quote_identifier
_cast_if_boolean = cast_if_boolean


def _normalize_variable_name(name: str) -> str:
    """Normalize variable name, converting special cases like '1' to '_intercept'"""
    if name == '1':
        return '_intercept'
    return name


def _make_sql_safe_name(name: str) -> str:
    """Convert a variable name/expression to a SQL-safe identifier.
    
    Removes special characters, replaces spaces/operators with underscores.
    Examples:
        'log(x+0.01)' -> 'log_x_0_01'
        'country*year' -> 'country_year'
        'L.gdp' -> 'L_gdp'
    """
    if name == '_intercept':
        return '_intercept'
    
    # Replace common patterns
    safe = name.replace('(', '_').replace(')', '').replace('+', '_').replace('-', '_')
    safe = safe.replace('*', '_').replace(':', '_').replace('.', '_').replace(' ', '_')
    safe = safe.replace('/', '_').replace(',', '_').replace(';', '_')
    
    # Remove consecutive underscores
    while '__' in safe:
        safe = safe.replace('__', '_')
    
    # Remove leading/trailing underscores
    safe = safe.strip('_')
    
    # Ensure it doesn't start with a digit
    if safe and safe[0].isdigit():
        safe = 'x_' + safe
    
    return safe


@dataclass(frozen=True)
class Variable:
    """Immutable representation of a parsed variable"""
    name: str
    role: VariableRole
    transform: TransformType = TransformType.NONE
    transform_shift: float = 0.0
    lag: Optional[int] = None
    display_name: Optional[str] = None
    sql_name: Optional[str] = None  # Clean SQL-safe identifier
    
    def __post_init__(self):
        # Build display name from transform info
        if self.display_name is None:
            display = self.name
            if self.transform != TransformType.NONE:
                if self.transform_shift != 0:
                    sign = '+' if self.transform_shift > 0 else ''
                    display = f"{self.transform.value}({self.name}{sign}{self.transform_shift})"
                else:
                    display = f"{self.transform.value}({self.name})"
            if self.lag is not None:
                prefix = 'L' if self.lag < 0 else 'F'
                offset = abs(self.lag)
                display = f"{prefix}{offset}.{display}" if offset > 1 else f"{prefix}.{display}"
            object.__setattr__(self, 'display_name', display)
        
        # Generate SQL-safe name if not provided
        if self.sql_name is None:
            object.__setattr__(self, 'sql_name', _make_sql_safe_name(self.display_name))
    
    def is_intercept(self) -> bool:
        """Check if this variable represents an intercept (constant 1)"""
        return self.name == '_intercept'
    
    def get_sql_expression(self, unit_col: Optional[str] = None, 
                          time_col: str = 'year') -> str:
        """Generate SQL expression for this variable"""
        # Special case for intercept
        if self.is_intercept():
            return "1"
        
        # Build base expression with proper quoting
        base_expr = quote_identifier(self.name)
        
        # Apply transformation if needed
        if self.transform == TransformType.NONE:
            expr = base_expr
        else:
            # For transforms, work with the quoted name
            inner = base_expr
            if self.transform_shift != 0:
                inner = f"({base_expr} + {self.transform_shift})"
            expr = self.transform.to_sql(inner)
        
        # Apply lag/lead if needed
        if self.lag is not None:
            partition = f"PARTITION BY {quote_identifier(unit_col)} " if unit_col else ""
            over_clause = f"OVER ({partition}ORDER BY {quote_identifier(time_col)})"
            func = "LAG" if self.lag < 0 else "LEAD"
            offset = abs(self.lag)
            expr = f"{func}({expr}, {offset}) {over_clause}"
        
        return expr
    
    def get_quoted_name(self) -> str:
        """Get the SQL-safe name (no quoting needed)"""
        return self.sql_name
    
    def get_quoted_display_name(self) -> str:
        """Get the SQL-safe name for use in SQL (no quoting needed)"""
        return self.sql_name
    
    def get_select_sql(self, unit_col: Optional[str] = None, 
                       time_col: str = 'year',
                       boolean_cols: Set[str] = None) -> str:
        """Generate SELECT clause fragment: expression AS sql_name"""
        expr = self.get_sql_expression(unit_col, time_col)
        expr = cast_if_boolean(expr, self.name, boolean_cols or set())
        return f"{expr} AS {self.sql_name}"


@dataclass(frozen=True)
class Interaction:
    """Immutable representation of an interaction term"""
    var1: Variable
    var2: Variable
    name: str
    sql_name: str  # Clean SQL-safe identifier
    include_main_effects: bool = True
    
    def get_quoted_name(self) -> str:
        """Get the SQL-safe name (no quoting needed)"""
        return self.sql_name
    
    def get_select_sql(self, unit_col: Optional[str] = None,
                       time_col: str = 'year',
                       boolean_cols: Set[str] = None) -> str:
        """Generate SELECT clause fragment: expression AS sql_name"""
        expr = self.get_sql_expression(unit_col, time_col, boolean_cols)
        return f"{expr} AS {self.sql_name}"


@dataclass(frozen=True)
class MergedFixedEffect:
    """Immutable representation of a merged fixed effect"""
    name: str
    sql_name: str  # Clean SQL-safe identifier
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
    
    def get_quoted_name(self) -> str:
        """Get the SQL-safe name (no quoting needed)"""
        return self.sql_name
    
    def get_select_sql(self, boolean_cols: Set[str] = None) -> str:
        """Generate SELECT clause fragment: expression AS sql_name"""
        return f"{self.get_sql_expression(boolean_cols)} AS {self.sql_name}"

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
    endogenous: Tuple[Variable, ...] = ()
    instruments: Tuple[Variable, ...] = ()
    
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
        return [var.display_name for var in self.outcomes]
    
    def get_covariate_names(self) -> List[str]:
        """Get all covariate names including interactions"""
        return [var.display_name for var in self.covariates] + [i.name for i in self.interactions]
    
    def get_simple_covariate_names(self) -> List[str]:
        """Get only simple covariate names (no interactions)"""
        return [var.display_name for var in self.covariates]
    
    def get_non_intercept_simple_covariate_names(self) -> List[str]:
        """Get simple covariate names excluding intercept"""
        return [var.display_name for var in self.covariates if not var.is_intercept()]
    
    def get_fe_names(self) -> List[str]:
        """Get all FE names including merged FEs"""
        return [var.display_name for var in self.fixed_effects] + [mfe.name for mfe in self.merged_fes]
    
    def get_simple_fe_names(self) -> List[str]:
        """Get only simple FE names (no merged FEs)"""
        return [var.display_name for var in self.fixed_effects]
    
    def get_endogenous_names(self) -> List[str]:
        """Get names of endogenous variables"""
        return [var.display_name for var in self.endogenous]
    
    def get_instrument_names(self) -> List[str]:
        """Get names of instrumental variables"""
        return [var.display_name for var in self.instruments]
    
    def get_exogenous_covariate_names(self) -> List[str]:
        """Get covariate names excluding endogenous variables"""
        endogenous_names = set(self.get_endogenous_names())
        return [name for name in self.get_covariate_names() if name not in endogenous_names]
    
    def has_instruments(self) -> bool:
        """Check if formula has instrumental variables"""
        return len(self.instruments) > 0
    
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
        
        # Add names from simple variables (excluding intercept)
        for var_list in (self.outcomes, self.covariates, self.fixed_effects, self.endogenous, self.instruments):
            for var in var_list:
                if not var.is_intercept():
                    cols.add(var.name)
        
        # Interaction components
        for interaction in self.interactions:
            if not interaction.var1.is_intercept():
                cols.add(interaction.var1.name)
            if not interaction.var2.is_intercept():
                cols.add(interaction.var2.name)
        
        # Merged FE components
        for mfe in self.merged_fes:
            cols.update(comp.name for comp in mfe.components if not comp.is_intercept())
        
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
            expr = cast_if_boolean(var.get_sql_expression(), var.name, boolean_cols)
            alias = quote_identifier(var.name)
            parts.append(f"{expr} AS {alias}")
        
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
        
        expr = cast_if_boolean(quote_identifier(col), col, boolean_cols)
        return f"{expr} AS {alias}"
    
    def get_where_clause_sql(self, user_subset: Optional[str] = None) -> str:
        """Generate complete WHERE clause with NULL check and optional user subset"""
        cols = self.get_source_columns_for_null_check()
        # Use column names as strings in COLUMNS() function
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
            return cast_if_boolean(expr, cov_var.name, boolean_cols)
        
        # Fallback: return quoted name
        return quote_identifier(cov_name)
    
    def get_fe_expression(self, fe_name: str, boolean_cols: Set[str] = None) -> str:
        """Get SQL expression for an FE by name (handles merged FEs)"""
        boolean_cols = boolean_cols or set()
        
        mfe = self.get_merged_fe_by_name(fe_name)
        if mfe:
            return mfe.get_sql_expression(boolean_cols)
        
        fe_var = self.get_fe_by_name(fe_name)
        if fe_var:
            expr = fe_var.get_sql_expression()
            return cast_if_boolean(expr, fe_var.name, boolean_cols)
        
        # Fallback: return quoted name
        return quote_identifier(fe_name)
    
    def get_instruments_select_sql(self, unit_col: Optional[str] = None,
                                   time_col: str = 'year',
                                   boolean_cols: Set[str] = None) -> str:
        """Generate SELECT clause for instrumental variables"""
        return self._join_select_parts([
            var.get_select_sql(unit_col, time_col, boolean_cols)
            for var in self.instruments
        ])
    
    def get_endogenous_select_sql(self, unit_col: Optional[str] = None,
                                  time_col: str = 'year',
                                  boolean_cols: Set[str] = None) -> str:
        """Generate SELECT clause for endogenous variables"""
        return self._join_select_parts([
            var.get_select_sql(unit_col, time_col, boolean_cols)
            for var in self.endogenous
        ])


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
        
        # Parse instrumental variables (part 3)
        endogenous, instruments = [], []
        if len(parts) > 2 and parts[2].strip() not in ("", "0"):
            endogenous, instruments = self._parse_instruments(parts[2])
        
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
            raw_formula=formula,
            endogenous=tuple(endogenous),
            instruments=tuple(instruments),
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
                    
                    interaction_name = f"{var1.name}_x_{var2.name}"
                    interaction_sql_name = f"{var1.sql_name}_x_{var2.sql_name}"
                    interactions.append(Interaction(
                        var1, var2, interaction_name, interaction_sql_name, include_main
                    ))
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
                merged_sql_name = '_'.join(c.sql_name for c in components)
                merged_fes.append(MergedFixedEffect(
                    name=merged_name,
                    sql_name=merged_sql_name,
                    components=components,
                    use_numeric_merge=(len(components) == 2)
                ))
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
            var_name = _normalize_variable_name(var_name)
            transform = TransformType[func_name] if func_name in TransformType.__members__ else TransformType.NONE
            return Variable(name=var_name, role=role, transform=transform, transform_shift=shift, lag=lag_val)
        
        # Normalize variable name (e.g., '1' -> '_intercept')
        normalized_name = _normalize_variable_name(term_no_lag)
        return Variable(name=normalized_name, role=role, lag=lag_val)
    
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
    
    def _parse_instruments(self, iv_string: str) -> Tuple[List[Variable], List[Variable]]:
        """Parse instrumental variables specification: endogenous(instrument1, instrument2, ...)
        
        Format: endog1 + endog2 (inst1 + inst2 + inst3)
        Or simpler: endog (inst1 + inst2)
        
        Handles functions like log(x + 0.01) correctly by finding the last opening paren
        that starts the instrument list.
        """
        iv_string = iv_string.strip()
        
        # Find the instrument parentheses - we need to find the opening paren that
        # corresponds to the instrument list, not function parentheses
        # The instrument list paren should be preceded by whitespace or end of variable name
        
        paren_start = self._find_instrument_paren_start(iv_string)
        
        if paren_start == -1 or ')' not in iv_string[paren_start:]:
            raise ValueError(
                "IV specification must be in format: endogenous (instrument1 + instrument2). "
                f"Got: {iv_string}"
            )
        
        paren_end = iv_string.rindex(')')
        
        endog_part = iv_string[:paren_start].strip()
        inst_part = iv_string[paren_start + 1:paren_end].strip()
        
        if not endog_part or not inst_part:
            raise ValueError(
                "IV specification requires both endogenous variables and instruments. "
                f"Got endogenous: '{endog_part}', instruments: '{inst_part}'"
            )
        
        endogenous = self._parse_variable_list(endog_part, VariableRole.ENDOGENOUS)
        instruments = self._parse_variable_list(inst_part, VariableRole.INSTRUMENT)
        
        if len(instruments) < len(endogenous):
            raise ValueError(
                f"Number of instruments ({len(instruments)}) must be at least "
                f"equal to number of endogenous variables ({len(endogenous)})"
            )
        
        return endogenous, instruments
    
    def _find_instrument_paren_start(self, iv_string: str) -> int:
        """Find the opening parenthesis that starts the instrument list.
        
        The instrument list paren is distinguished from function parens by:
        1. It's preceded by whitespace (after the endogenous variables)
        2. OR it's at a position where parentheses are balanced before it
        
        Examples:
        - "log(x + 0.01) (inst1 + inst2)" -> returns index of " (" after ")"
        - "x (inst1)" -> returns index of " (" after "x"
        - "log(x) + y (inst1 + inst2)" -> returns index of " (" after "y"
        """
        # Track parenthesis depth as we scan
        depth = 0
        last_balanced_pos = -1
        
        for i, char in enumerate(iv_string):
            if char == '(':
                # Check if this could be the instrument paren
                # It should be preceded by whitespace when depth is 0
                if depth == 0 and i > 0:
                    # Check if preceded by whitespace
                    prev_char = iv_string[i - 1]
                    if prev_char.isspace():
                        # This is likely the instrument list paren
                        return i
                depth += 1
            elif char == ')':
                depth -= 1
                if depth == 0:
                    last_balanced_pos = i
        
        # If we didn't find a paren preceded by whitespace at depth 0,
        # look for the last opening paren that comes after balanced parens
        depth = 0
        for i, char in enumerate(iv_string):
            if char == '(':
                if depth == 0 and i > last_balanced_pos and last_balanced_pos >= 0:
                    # Found an opening paren after all previous parens are balanced
                    return i
                depth += 1
            elif char == ')':
                depth -= 1
        
        # Fallback: find first '(' preceded by space, or first '(' if no functions
        for i, char in enumerate(iv_string):
            if char == '(' and i > 0 and iv_string[i-1].isspace():
                return i
        
        # Last resort: if there's only one set of parens and no functions, use it
        if '(' in iv_string:
            first_paren = iv_string.index('(')
            # Check if this looks like a function call (preceded by alphanumeric)
            if first_paren > 0 and iv_string[first_paren - 1].isalnum():
                # This is a function call, look for the next paren
                rest = iv_string[first_paren + 1:]
                depth = 1
                for i, char in enumerate(rest):
                    if char == '(':
                        depth += 1
                    elif char == ')':
                        depth -= 1
                        if depth == 0:
                            # Found end of function, look for next paren
                            remaining = rest[i + 1:]
                            if '(' in remaining:
                                return first_paren + 1 + i + 1 + remaining.index('(')
                            break
            else:
                return first_paren
        
        return -1
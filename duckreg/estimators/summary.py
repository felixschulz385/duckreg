"""
Summary formatting for regression and 2SLS results.

This module provides comprehensive formatting for model output, including:
- Model specification (estimator, variables, FE, clustering)
- Sample information with compression ratios
- Coefficient results with significance indicators
- First stage diagnostics for IV models
"""

import pandas as pd
from typing import Union, Optional, Dict, Any

from .results import RegressionResults, FirstStageResults


def format_model_summary(
    model_summary: Dict[str, Any],
    spec_config: Optional[Dict[str, Any]] = None,
    precision: int = 4
) -> str:
    """Format comprehensive model summary for printing and storage.
    
    This provides the most useful printable representation including:
    - Model specification (estimator, variables, FE, clustering)
    - Sample information (n_obs, compression ratio)
    - Coefficient results with significance
    - First stage diagnostics for IV models
    
    Args:
        model_summary: Model summary dictionary from estimator.summary()
        spec_config: Optional specification config with description
        precision: Number of decimal places for display
        
    Returns:
        Formatted string suitable for console output and file storage
    """
    lines = []
    
    # Header
    lines.append("=" * 80)
    if spec_config and 'description' in spec_config:
        lines.append(f"ANALYSIS: {spec_config['description']}")
    else:
        lines.append("REGRESSION ANALYSIS RESULTS")
    lines.append("=" * 80)
    
    # Version and timestamp
    version_info = model_summary.get('version_info', {})
    if version_info:
        lines.append(f"\nDuckReg Version: {version_info.get('duckreg_version', 'unknown')}")
        lines.append(f"Computed at: {version_info.get('computed_at', 'unknown')}")
    
    # Model specification
    model_spec = model_summary.get('model_spec', {})
    if model_spec:
        lines.append("\nMODEL SPECIFICATION")
        lines.append("-" * 80)
        lines.append(f"Estimator: {model_spec.get('estimator_type', 'unknown')}")
        
        outcome_vars = model_spec.get('outcome_vars', [])
        if outcome_vars:
            lines.append(f"Outcome: {', '.join(outcome_vars)}")
        
        covariate_vars = model_spec.get('covariate_vars', [])
        if covariate_vars:
            lines.append(f"Covariates: {', '.join(covariate_vars)}")
        
        # IV-specific variables
        if model_spec.get('endogenous_vars'):
            lines.append(f"Endogenous: {', '.join(model_spec.get('endogenous_vars', []))}")
        if model_spec.get('instrument_vars'):
            lines.append(f"Instruments: {', '.join(model_spec.get('instrument_vars', []))}")
        if model_spec.get('exogenous_vars'):
            lines.append(f"Exogenous: {', '.join(model_spec.get('exogenous_vars', []))}")
        
        # Fixed effects
        fe_cols = model_spec.get('fe_cols', [])
        if fe_cols:
            lines.append(f"Fixed Effects: {', '.join(fe_cols)}")
            fe_method = model_spec.get('fe_method')
            if fe_method:
                lines.append(f"FE Method: {fe_method}")
        
        # Clustering
        cluster_col = model_spec.get('cluster_col')
        if cluster_col:
            lines.append(f"Clustering: {cluster_col}")
    
    # Sample information
    sample_info = model_summary.get('sample_info', {})
    if sample_info:
        lines.append("\nSAMPLE INFORMATION")
        lines.append("-" * 80)
        
        n_obs = sample_info.get('n_obs')
        if n_obs is not None:
            lines.append(f"Observations: {n_obs:,}")
        
        n_compressed = sample_info.get('n_compressed')
        if n_compressed is not None:
            lines.append(f"Compressed Rows: {n_compressed:,}")
        
        compression_ratio = sample_info.get('compression_ratio')
        if compression_ratio is not None and n_obs and n_compressed:
            lines.append(f"Compression: {compression_ratio:.1%} reduction ({n_obs:,} → {n_compressed:,} rows)")
    
    # Coefficient results
    coefficients = model_summary.get('coefficients')
    if coefficients and isinstance(coefficients, dict):
        lines.append("\nCOEFFICIENT RESULTS")
        lines.append("-" * 80)
        
        coef_names = coefficients.get('coef_names', [])
        coefs = coefficients.get('coefficients', [])
        ses = coefficients.get('std_errors')
        t_stats = coefficients.get('t_statistics')
        p_vals = coefficients.get('p_values')
        se_type = coefficients.get('se_type')
        
        if se_type:
            lines.append(f"Standard Errors: {se_type}")
            lines.append("")
        
        if coef_names and coefs:
            # Table header
            if ses and t_stats and p_vals:
                fmt_str = f"{{:<30}} {{:>12}} {{:>12}} {{:>10}} {{:>10}} {{:>3}}"
                header = fmt_str.format("Variable", "Coefficient", "Std. Error", "t-stat", "p-value", "")
            else:
                fmt_str = f"{{:<30}} {{:>12}}"
                header = fmt_str.format("Variable", "Coefficient")
            
            lines.append(header)
            lines.append("-" * 80)
            
            # Coefficient rows
            for i, name in enumerate(coef_names):
                coef = coefs[i]
                sig = ""
                
                if ses and t_stats and p_vals:
                    se = ses[i]
                    t = t_stats[i]
                    p = p_vals[i]
                    
                    # Significance stars
                    if p < 0.001:
                        sig = "***"
                    elif p < 0.01:
                        sig = "**"
                    elif p < 0.05:
                        sig = "*"
                    elif p < 0.10:
                        sig = "."
                    
                    line = fmt_str.format(
                        name[:29],
                        f"{coef:.{precision}f}",
                        f"{se:.{precision}f}",
                        f"{t:.{precision}f}",
                        f"{p:.{precision}f}",
                        sig
                    )
                else:
                    line = fmt_str.format(name[:29], f"{coef:.{precision}f}")
                
                lines.append(line)
            
            lines.append("-" * 80)
            if ses and t_stats and p_vals:
                lines.append("Significance: *** p<0.001, ** p<0.01, * p<0.05, . p<0.10")
    
    # First stage diagnostics for IV models
    first_stage = model_summary.get('first_stage')
    iv_diagnostics = model_summary.get('iv_diagnostics')
    
    if first_stage and iv_diagnostics:
        lines.append("\n" + "=" * 80)
        lines.append("FIRST STAGE DIAGNOSTICS (IV/2SLS)")
        lines.append("=" * 80)
        
        weak_instruments = iv_diagnostics.get('weak_instruments')
        if weak_instruments is not None:
            status = "YES ⚠️" if weak_instruments else "NO"
            lines.append(f"\nWeak Instruments: {status}")
        
        f_stats = iv_diagnostics.get('first_stage_f_stats', {})
        if f_stats:
            lines.append("\nFirst Stage F-Statistics:")
            lines.append("-" * 80)
            for endog, f_stat in f_stats.items():
                if f_stat is not None:
                    status = " [WEAK]" if f_stat < 10 else ""
                    lines.append(f"{endog:30s}: F = {f_stat:10.2f}{status}")
        
        # Show detailed first stage results
        if first_stage:
            lines.append("\nFirst Stage Coefficient Results:")
            for endog, fs_dict in first_stage.items():
                lines.append("\n" + "-" * 80)
                lines.append(f"Endogenous Variable: {endog}")
                lines.append("-" * 80)
                
                if isinstance(fs_dict, dict) and 'coef_names' in fs_dict:
                    fs_coef_names = fs_dict.get('coef_names', [])
                    fs_coefs = fs_dict.get('coefficients', [])
                    fs_ses = fs_dict.get('std_errors')
                    fs_t_stats = fs_dict.get('t_statistics')
                    fs_p_vals = fs_dict.get('p_values')
                    
                    if fs_coef_names and fs_coefs:
                        fmt_str = f"{{:<25}} {{:>12}} {{:>12}} {{:>10}} {{:>10}}"
                        header = fmt_str.format("Variable", "Coefficient", "Std. Error", "t-stat", "p-value")
                        lines.append(header)
                        
                        for i, name in enumerate(fs_coef_names):
                            coef = fs_coefs[i]
                            if fs_ses and fs_t_stats and fs_p_vals:
                                line = fmt_str.format(
                                    name[:24],
                                    f"{coef:.{precision}f}",
                                    f"{fs_ses[i]:.{precision}f}",
                                    f"{fs_t_stats[i]:.{precision}f}",
                                    f"{fs_p_vals[i]:.{precision}f}"
                                )
                            else:
                                line = f"{name[:24]:<25} {coef:>12.{precision}f}"
                            lines.append(line)
    
    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


# Backward compatibility: convenience functions and SummaryFormatter class
def format_summary(
    result: Union[RegressionResults, FirstStageResults, Dict[str, Any]],
    precision: int = 4,
    include_diagnostics: bool = True
) -> str:
    """Format results for console output."""
    if isinstance(result, dict):
        return format_model_summary(result, precision=precision)
    else:
        return str(result)


def print_summary(
    result: Union[RegressionResults, FirstStageResults, Dict[str, Any]],
    precision: int = 4,
    include_diagnostics: bool = True
):
    """Print formatted results to console."""
    print(format_summary(result, precision, include_diagnostics))


def to_tidy_df(
    result: Union[RegressionResults, FirstStageResults, Dict[str, Any]]
) -> pd.DataFrame:
    """Convert results to tidy DataFrame."""
    if isinstance(result, (RegressionResults, FirstStageResults)):
        return result.to_tidy_df()
    return pd.DataFrame()


class SummaryFormatter:
    """Backward-compatible summary formatter (use format_model_summary() instead)."""
    
    @staticmethod
    def format(
        result: Union[RegressionResults, FirstStageResults, Dict[str, Any]],
        precision: int = 4,
        include_diagnostics: bool = True
    ) -> str:
        """Format results from dict or result objects."""
        return format_summary(result, precision, include_diagnostics)
    
    @staticmethod
    def print(
        result: Union[RegressionResults, FirstStageResults, Dict[str, Any]],
        precision: int = 4,
        include_diagnostics: bool = True
    ):
        """Print formatted results to console."""
        print_summary(result, precision, include_diagnostics)
    
    @staticmethod
    def to_tidy_df(
        result: Union[RegressionResults, FirstStageResults, Dict[str, Any]]
    ) -> pd.DataFrame:
        """Convert results to tidy DataFrame."""
        return to_tidy_df(result)



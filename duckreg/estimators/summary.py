"""
Unified summary formatting for regression and 2SLS results.

This module provides a smart summary formatter that:
- Detects result type and formats appropriately
- Supports both text (console) and tidy DataFrame outputs
- Unifies the interface across DuckRegression, DuckMundlak, and Duck2SLS
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any, List

from .results import RegressionResults, FirstStageResults


class SummaryFormatter:
    """Unified formatter for regression and 2SLS results."""
    
    @staticmethod
    def format(
        result: Union[RegressionResults, FirstStageResults, Dict[str, Any]],
        precision: int = 4,
        include_diagnostics: bool = True
    ) -> str:
        """
        Format results for console output (smart detection of result type).
        
        Args:
            result: RegressionResults, FirstStageResults, or summary dict
            precision: Number of decimal places
            include_diagnostics: Include diagnostic information (2SLS only)
            
        Returns:
            Formatted string for printing
        """
        if isinstance(result, FirstStageResults):
            return _format_first_stage(result, precision)
        elif isinstance(result, RegressionResults):
            return _format_regression(result, precision)
        elif isinstance(result, dict) and 'first_stage' in result:
            return _format_2sls_summary(result, precision, include_diagnostics)
        elif isinstance(result, dict):
            return _format_generic_summary(result)
        else:
            return str(result)
    
    @staticmethod
    def to_tidy_df(
        result: Union[RegressionResults, FirstStageResults, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Convert results to tidy DataFrame (smart detection of result type).
        
        Args:
            result: RegressionResults, FirstStageResults, or summary dict
            
        Returns:
            Tidy (long format) DataFrame
        """
        if isinstance(result, (RegressionResults, FirstStageResults)):
            return result.to_tidy_df()
        elif isinstance(result, dict) and 'coefficients' in result:
            # Try to reconstruct from dict
            coef_dict = result.get('coefficients', {})
            if isinstance(coef_dict, dict) and 'coef_names' in coef_dict:
                return _dict_to_tidy_df(coef_dict)
        return pd.DataFrame()
    
    @staticmethod
    def print(
        result: Union[RegressionResults, FirstStageResults, Dict[str, Any]],
        precision: int = 4,
        include_diagnostics: bool = True
    ):
        """Print formatted results to console."""
        print(SummaryFormatter.format(result, precision, include_diagnostics))


def _format_regression(results: RegressionResults, precision: int = 4) -> str:
    """Format standard regression results."""
    return results.to_string(precision=precision)


def _format_first_stage(results: FirstStageResults, precision: int = 4) -> str:
    """Format first-stage regression results with F-statistic."""
    return results.to_string(precision=precision)


def _format_2sls_summary(summary_dict: Dict[str, Any], precision: int = 4, 
                         include_diagnostics: bool = True) -> str:
    """Format comprehensive 2SLS summary with first stages and diagnostics."""
    lines = []
    lines.append("=" * 90)
    lines.append("TWO-STAGE LEAST SQUARES (2SLS) RESULTS")
    lines.append("=" * 90)
    
    # Model specification
    if 'exogenous_vars' in summary_dict or 'endogenous_vars' in summary_dict:
        lines.append("\nMODEL SPECIFICATION")
        lines.append("-" * 90)
        if 'outcome_vars' in summary_dict:
            lines.append(f"Outcome:         {', '.join(summary_dict['outcome_vars'])}")
        if 'exogenous_vars' in summary_dict:
            lines.append(f"Exogenous:       {', '.join(summary_dict['exogenous_vars']) or 'None'}")
        if 'endogenous_vars' in summary_dict:
            lines.append(f"Endogenous:      {', '.join(summary_dict['endogenous_vars'])}")
        if 'instrument_vars' in summary_dict:
            lines.append(f"Instruments:     {', '.join(summary_dict['instrument_vars'])}")
        if 'fe_cols' in summary_dict:
            lines.append(f"Fixed Effects:   {', '.join(summary_dict['fe_cols']) or 'None'}")
        if 'fe_method' in summary_dict and summary_dict['fe_method']:
            lines.append(f"FE Method:       {summary_dict['fe_method']}")
        if 'cluster_col' in summary_dict and summary_dict['cluster_col']:
            lines.append(f"Clustering:      {summary_dict['cluster_col']}")
    
    # Observations
    if 'n_obs' in summary_dict:
        lines.append("\nSAMPLE INFORMATION")
        lines.append("-" * 90)
        lines.append(f"Observations:    {summary_dict['n_obs']:,}")
        if 'n_compressed' in summary_dict:
            lines.append(f"Compressed:      {summary_dict['n_compressed']:,}")
    
    # First stage diagnostics
    if include_diagnostics and 'first_stage' in summary_dict and summary_dict['first_stage']:
        lines.append("\nFIRST STAGE DIAGNOSTICS")
        lines.append("-" * 90)
        for endog, fs_dict in summary_dict['first_stage'].items():
            f_stat = fs_dict.get('f_statistic')
            f_pval = fs_dict.get('f_pvalue')
            is_weak = fs_dict.get('is_weak_instrument')
            
            if f_stat is not None:
                status = " [WEAK INSTRUMENTS]" if is_weak else ""
                lines.append(f"{endog:20s}: F = {f_stat:8.{precision}f}, "
                            f"p-value = {f_pval:.{precision}f}{status}")
        
        if summary_dict.get('weak_instruments'):
            lines.append("\n⚠️  WARNING: Weak instruments detected (F < 10)")
    
    # Second stage coefficients
    if 'coefficients' in summary_dict and summary_dict['coefficients']:
        coef_dict = summary_dict['coefficients']
        if isinstance(coef_dict, dict):
            lines.append("\nSECOND STAGE RESULTS")
            lines.append("-" * 90)
            
            # Extract and format coefficient table
            coef_names = coef_dict.get('coef_names', [])
            coefs = coef_dict.get('coefficients', [])
            ses = coef_dict.get('std_errors')
            t_stats = coef_dict.get('t_statistics')
            p_vals = coef_dict.get('p_values')
            
            if coef_names and coefs:
                fmt_str = f"{{:<25}} {{:>12}} {{:>12}} {{:>10}} {{:>10}} {{:>2}}"
                header = fmt_str.format("Variable", "Coefficient", "Std. Error", "t-stat", "p-value", "")
                lines.append(header)
                lines.append("-" * 90)
                
                for i, name in enumerate(coef_names):
                    coef = coefs[i]
                    sig = ""
                    
                    if ses and t_stats and p_vals:
                        se = ses[i]
                        t = t_stats[i]
                        p = p_vals[i]
                        
                        if p < 0.001:
                            sig = "***"
                        elif p < 0.01:
                            sig = "**"
                        elif p < 0.05:
                            sig = "*"
                        elif p < 0.10:
                            sig = "."
                        
                        line = fmt_str.format(
                            name[:24],
                            f"{coef:.{precision}f}",
                            f"{se:.{precision}f}",
                            f"{t:.{precision}f}",
                            f"{p:.{precision}f}",
                            sig
                        )
                    else:
                        line = fmt_str.format(name[:24], f"{coef:.{precision}f}", "", "", "")
                    
                    lines.append(line)
                
                lines.append("-" * 90)
                lines.append("Significance codes: *** p<0.001, ** p<0.01, * p<0.05, . p<0.10")
    
    lines.append("=" * 90)
    return "\n".join(lines)


def _format_generic_summary(summary_dict: Dict[str, Any]) -> str:
    """Format generic dictionary summary."""
    lines = ["SUMMARY", "=" * 80]
    for key, value in summary_dict.items():
        if isinstance(value, (dict, list)) and len(str(value)) > 100:
            lines.append(f"{key}: [nested structure]")
        else:
            lines.append(f"{key}: {value}")
    lines.append("=" * 80)
    return "\n".join(lines)


def _dict_to_tidy_df(coef_dict: Dict[str, Any]) -> pd.DataFrame:
    """Convert coefficient dictionary to tidy DataFrame."""
    data = {
        'variable': coef_dict.get('coef_names', []),
        'estimate': coef_dict.get('coefficients', []),
    }
    
    if 'std_errors' in coef_dict:
        data['std_error'] = coef_dict['std_errors']
    if 't_statistics' in coef_dict:
        data['t_stat'] = coef_dict['t_statistics']
    if 'p_values' in coef_dict:
        data['p_value'] = coef_dict['p_values']
    
    return pd.DataFrame(data) if data.get('variable') else pd.DataFrame()


# Convenience functions for direct access
def format_summary(
    result: Union[RegressionResults, FirstStageResults, Dict[str, Any]],
    precision: int = 4,
    include_diagnostics: bool = True
) -> str:
    """Format results for console output."""
    return SummaryFormatter.format(result, precision, include_diagnostics)


def print_summary(
    result: Union[RegressionResults, FirstStageResults, Dict[str, Any]],
    precision: int = 4,
    include_diagnostics: bool = True
):
    """Print formatted results."""
    SummaryFormatter.print(result, precision, include_diagnostics)


def to_tidy_df(
    result: Union[RegressionResults, FirstStageResults, Dict[str, Any]]
) -> pd.DataFrame:
    """Convert to tidy DataFrame."""
    return SummaryFormatter.to_tidy_df(result)

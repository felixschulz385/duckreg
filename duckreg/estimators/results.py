"""
Result containers for regression output.

This module follows Single Responsibility Principle - contains only
data containers for storing regression results.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple


@dataclass
class RegressionResults:
    """Container for regression results with computed statistics"""
    coefficients: np.ndarray
    coef_names: List[str]
    vcov: Optional[np.ndarray] = None
    n_obs: Optional[int] = None
    n_compressed: Optional[int] = None
    se_type: Optional[str] = None
    
    @property
    def std_errors(self) -> Optional[np.ndarray]:
        if self.vcov is None:
            return None
        return np.sqrt(np.diag(self.vcov))
    
    @property
    def t_stats(self) -> Optional[np.ndarray]:
        se = self.std_errors
        if se is None:
            return None
        return self.coefficients.flatten() / se
    
    @property
    def p_values(self) -> Optional[np.ndarray]:
        t = self.t_stats
        if t is None:
            return None
        from scipy import stats
        return 2 * (1 - stats.norm.cdf(np.abs(t)))
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        data = {'coefficient': self.coefficients.flatten()}
        if self.std_errors is not None:
            data['std_error'] = self.std_errors
            data['t_stat'] = self.t_stats
            data['p_value'] = self.p_values
            data['ci_lower'] = self.coefficients.flatten() - 1.96 * self.std_errors
            data['ci_upper'] = self.coefficients.flatten() + 1.96 * self.std_errors
        return pd.DataFrame(data, index=self.coef_names)
    
    def to_tidy_df(self) -> pd.DataFrame:
        """Convert to tidy (long format) DataFrame.
        
        Each coefficient becomes a row with variable name, estimate, and stats.
        Useful for plotting and data analysis.
        
        Returns:
            DataFrame with columns: variable, estimate, std_error, t_stat, p_value, ci_lower, ci_upper
        """
        data = {
            'variable': self.coef_names,
            'estimate': self.coefficients.flatten(),
        }
        if self.std_errors is not None:
            data['std_error'] = self.std_errors
            data['t_stat'] = self.t_stats
            data['p_value'] = self.p_values
            data['ci_lower'] = self.coefficients.flatten() - 1.96 * self.std_errors
            data['ci_upper'] = self.coefficients.flatten() + 1.96 * self.std_errors
        return pd.DataFrame(data)
    
    def to_string(self, precision: int = 4) -> str:
        """Generate a printable text summary of results.
        
        Args:
            precision: Number of decimal places to display
            
        Returns:
            Formatted string suitable for printing/console output
        """
        lines = []
        lines.append("=" * 80)
        lines.append("REGRESSION RESULTS")
        lines.append("=" * 80)
        
        if self.n_obs is not None:
            lines.append(f"Number of observations: {self.n_obs:,}")
        if self.n_compressed is not None:
            lines.append(f"Compressed rows: {self.n_compressed:,}")
        if self.se_type is not None:
            lines.append(f"Standard error type: {self.se_type}")
        
        lines.append("-" * 80)
        
        # Build results table
        if self.std_errors is not None:
            fmt_str = f"{{:<30}} {{:>12}} {{:>12}} {{:>10}} {{:>10}}"
            header = fmt_str.format("Variable", "Coefficient", "Std. Error", "t-stat", "p-value")
            lines.append(header)
            lines.append("-" * 80)
            
            for i, name in enumerate(self.coef_names):
                coef = self.coefficients.flatten()[i]
                se = self.std_errors[i]
                t_stat = self.t_stats[i]
                p_val = self.p_values[i]
                
                sig = ""
                if p_val < 0.001:
                    sig = "***"
                elif p_val < 0.01:
                    sig = "**"
                elif p_val < 0.05:
                    sig = "*"
                elif p_val < 0.10:
                    sig = "."
                
                line = fmt_str.format(
                    name[:29],
                    f"{coef:.{precision}f}",
                    f"{se:.{precision}f}",
                    f"{t_stat:.{precision}f}",
                    f"{p_val:.{precision}f}",
                    sig
                )
                lines.append(line)
            
            lines.append("-" * 80)
            lines.append("Significance codes: *** p<0.001, ** p<0.01, * p<0.05, . p<0.10")
        else:
            # No standard errors available
            lines.append(f"{'Variable':<30} {'Coefficient':>15}")
            lines.append("-" * 80)
            for i, name in enumerate(self.coef_names):
                coef = self.coefficients.flatten()[i]
                lines.append(f"{name:<30} {coef:>15.{precision}f}")
            lines.append("-" * 80)
        
        lines.append("=" * 80)
        return "\n".join(lines)
    
    def print_summary(self, precision: int = 4):
        """Print regression results summary to console.
        
        Args:
            precision: Number of decimal places to display
        """
        print(self.to_string(precision=precision))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            'coefficients': self.coefficients.flatten().tolist(),
            'coef_names': self.coef_names,
            'n_obs': self.n_obs,
            'n_compressed': self.n_compressed,
        }
        if self.vcov is not None:
            result.update({
                'vcov': self.vcov.tolist(),
                'std_errors': self.std_errors.tolist(),
                't_statistics': self.t_stats.tolist(),
                'p_values': self.p_values.tolist(),
                'se_type': self.se_type,
            })
        return result


@dataclass
class FirstStageResults:
    """Container for first-stage regression results with instrument diagnostics"""
    endog_var: str
    results: RegressionResults
    instrument_names: List[str]
    
    # Computed on demand
    _f_stat: Optional[float] = field(default=None, repr=False)
    _f_pvalue: Optional[float] = field(default=None, repr=False)
    
    @property
    def coefficients(self) -> np.ndarray:
        return self.results.coefficients
    
    @property
    def coef_names(self) -> List[str]:
        return self.results.coef_names
    
    @property
    def vcov(self) -> Optional[np.ndarray]:
        return self.results.vcov
    
    @property
    def n_obs(self) -> Optional[int]:
        return self.results.n_obs
    
    def compute_f_statistic(self) -> Tuple[Optional[float], Optional[float]]:
        """Compute F-statistic for joint significance of instruments"""
        if self.vcov is None:
            return None, None
        
        # Find instrument indices
        inst_indices = [i for i, name in enumerate(self.coef_names) if name in self.instrument_names]
        if not inst_indices:
            return None, None
        
        coefs = self.coefficients.flatten()
        inst_coefs = coefs[inst_indices]
        inst_vcov = self.vcov[np.ix_(inst_indices, inst_indices)]
        
        try:
            inst_vcov_inv = np.linalg.inv(inst_vcov)
            wald_stat = inst_coefs @ inst_vcov_inv @ inst_coefs
            n_inst = len(inst_indices)
            f_stat = wald_stat / n_inst
            
            # P-value
            from scipy import stats
            df1 = n_inst
            df2 = (self.n_obs or 1000) - len(coefs)
            f_pvalue = 1 - stats.f.cdf(f_stat, df1, df2)
            
            self._f_stat = float(f_stat)
            self._f_pvalue = float(f_pvalue)
            return self._f_stat, self._f_pvalue
        except np.linalg.LinAlgError:
            return None, None
    
    @property
    def f_statistic(self) -> Optional[float]:
        if self._f_stat is None:
            self.compute_f_statistic()
        return self._f_stat
    
    @property
    def f_pvalue(self) -> Optional[float]:
        if self._f_pvalue is None:
            self.compute_f_statistic()
        return self._f_pvalue
    
    @property
    def is_weak_instrument(self) -> Optional[bool]:
        """Check if F < 10 (Stock-Yogo rule of thumb)"""
        f = self.f_statistic
        return f < 10 if f is not None else None
    
    def get_instrument_stats(self) -> Dict[str, Dict[str, float]]:
        """Get individual instrument coefficient statistics"""
        if self.vcov is None:
            return {}
        
        coefs = self.coefficients.flatten()
        se = self.results.std_errors
        t_stats = self.results.t_stats
        p_vals = self.results.p_values
        
        stats = {}
        for i, name in enumerate(self.coef_names):
            if name in self.instrument_names:
                stats[name] = {
                    'coefficient': float(coefs[i]),
                    'std_error': float(se[i]),
                    't_statistic': float(t_stats[i]),
                    'p_value': float(p_vals[i]),
                }
        return stats
    
    def to_tidy_df(self) -> pd.DataFrame:
        """Convert to tidy (long format) DataFrame with instrument flags.
        
        Returns:
            DataFrame with columns: variable, estimate, std_error, t_stat, p_value, 
                                    ci_lower, ci_upper, is_instrument
        """
        df = self.results.to_tidy_df()
        df['is_instrument'] = df['variable'].isin(self.instrument_names)
        return df
    
    def to_string(self, precision: int = 4) -> str:
        """Generate a printable text summary of first-stage results.
        
        Args:
            precision: Number of decimal places to display
            
        Returns:
            Formatted string suitable for printing/console output
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"FIRST STAGE RESULTS: {self.endog_var}")
        lines.append("=" * 80)
        
        # Instrument diagnostics
        lines.append(f"F-statistic: {self.f_statistic:.{precision}f}")
        lines.append(f"P-value: {self.f_pvalue:.{precision}f}")
        if self.is_weak_instrument is not None:
            status = "WEAK (F < 10)" if self.is_weak_instrument else "Strong"
            lines.append(f"Instrument strength: {status}")
        
        lines.append("-" * 80)
        lines.append(self.results.to_string(precision=precision).split("=" * 80)[1])  # Extract middle part
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def print_summary(self, precision: int = 4):
        """Print first-stage results summary to console.
        
        Args:
            precision: Number of decimal places to display
        """
        print(self.to_string(precision=precision))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'endogenous_variable': self.endog_var,
            **self.results.to_dict(),
            'instrument_names': self.instrument_names,
            'f_statistic': self.f_statistic,
            'f_pvalue': self.f_pvalue,
            'is_weak_instrument': self.is_weak_instrument,
            'instrument_statistics': self.get_instrument_stats(),
        }

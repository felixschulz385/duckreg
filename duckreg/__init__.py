"""
.. include:: ../README.md
"""

from .duckreg import compressed_ols, DuckReg, wls
from .estimators import DuckRegression, DuckMundlak, DuckMundlakEventStudy, DuckDoubleDemeaning

__all__ = [
    "compressed_ols",
    "DuckReg",
    "DuckRegression",
    "DuckMundlak",
    "DuckMundlakEventStudy",
    "DuckDoubleDemeaning",
    "wls",
]

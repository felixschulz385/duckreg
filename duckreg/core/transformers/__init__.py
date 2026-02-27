"""Fixed-effects transformer package.

Transformers encapsulate data transformation logic for absorbing fixed effects,
decoupled from formula parsing and coefficient estimation.  They can be composed
into any estimator that needs FE absorption.

Available transformers
----------------------
FETransformer
    Abstract base class defining the transformer interface.
IterativeDemeanTransformer
    Absorbs FEs via the Method of Alternating Projections (MAP).
MundlakTransformer
    Absorbs FEs via the Mundlak device (group-mean regressors).
"""

from .base import FETransformer
from .iterative_demean import IterativeDemeanTransformer
from .mundlak import MundlakTransformer
from .auto_fe import AutoFETransformer

__all__ = [
    "FETransformer",
    "IterativeDemeanTransformer",
    "MundlakTransformer",
    "AutoFETransformer",
]

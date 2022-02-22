# __init__.py
# cleaning up namespace - sjs 1.27.22
try: # in stdlib since 3.8
    from importlib.metadata import distribution
    from importlib.metadata import PackageNotFoundError as VersionError
except ImportError: # for older versions
    from pkg_resources import get_distribution as distribution
    from pkg_resources import DistributionNotFound as VersionError
from .metrics import (TropD_Metric_EDJ,
                      TropD_Metric_OLR,
                      TropD_Metric_PE,
                      TropD_Metric_PSI,
                      TropD_Metric_PSL,
                      TropD_Metric_STJ,
                      TropD_Metric_TPB,
                      TropD_Metric_UAS)
from .functions import (TropD_Calculate_MaxLat,
                        TropD_Calculate_Mon2Season,
                        TropD_Calculate_StreamFunction,
                        TropD_Calculate_TropopauseHeight,
                        TropD_Calculate_ZeroCrossing)
                        
try:
    __version__ = distribution("pytropd").version
except VersionError:
    # Local copy or not installed with setuptools.
    #__version__ = "999"
    __version__ = '1.0.5'
    
__all__ = ("TropD_Calculate_MaxLat",
           "TropD_Calculate_Mon2Season",
           "TropD_Calculate_StreamFunction",
           "TropD_Calculate_TropopauseHeight",
           "TropD_Calculate_ZeroCrossing",
           "TropD_Metric_EDJ",
           "TropD_Metric_OLR",
           "TropD_Metric_PE",
           "TropD_Metric_PSI",
           "TropD_Metric_PSL",
           "TropD_Metric_STJ",
           "TropD_Metric_TPB",
           "TropD_Metric_UAS")

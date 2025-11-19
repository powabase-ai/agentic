"""ETL pipeline framework for document processing"""

from .pipeline import Pipeline
from .config import ETLConfig

# Import connectors to register them
from . import connectors  # noqa: F401

# Import extractors to register them
from . import transformers  # noqa: F401
from .transformers import extractors  # noqa: F401

# Import chunkers to register them
from .transformers import chunkers  # noqa: F401

# Import embedders to register them
from .transformers import embedders  # noqa: F401

# Import sinks to register them
from . import sinks  # noqa: F401

__all__ = ["Pipeline", "ETLConfig"]


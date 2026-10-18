"""Model components. Importing this package registers every encoder, world model and head."""
import importlib
import pkgutil

from wmpoc.models import encoders, world_models  # noqa: F401
from wmpoc.models import heads  # noqa: F401

for _pkg in (encoders, world_models):
    for _m in pkgutil.iter_modules(_pkg.__path__):
        importlib.import_module(f"{_pkg.__name__}.{_m.name}")

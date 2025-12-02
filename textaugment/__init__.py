import os
import importlib

name = "textaugment"

__version__ = "2.0.0"
__licence__ = "MIT"
__author__ = "Joseph Sefara"
__url__ = "https://github.com/dsfsi/textaugment/"

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

__all__ = [
    "Translate",
    "Word2vec",
    "Fasttext",
    "Wordnet",
    "EDA",
    "AEDA",
    "MIXUP",
    "LANGUAGES",
]

# Map attribute → module path → object name
_lazy_modules = {
    "Translate": ("textaugment.translate", "Translate"),
    "Word2vec": ("textaugment.word2vec", "Word2vec"),
    "Fasttext": ("textaugment.word2vec", "Fasttext"),
    "Wordnet": ("textaugment.wordnet", "Wordnet"),
    "EDA": ("textaugment.eda", "EDA"),
    "AEDA": ("textaugment.aeda", "AEDA"),
    "MIXUP": ("textaugment.mixup", "MIXUP"),
    "LANGUAGES": ("textaugment.constants", "LANGUAGES"),
}


def __getattr__(name: str):
    """Dynamically import modules when attributes are requested."""
    if name in _lazy_modules:
        module_name, attr_name = _lazy_modules[name]
        module = importlib.import_module(module_name)
        attr = getattr(module, attr_name)
        globals()[name] = attr  # cache to avoid reloading
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Ensure lazy attributes appear in dir()."""
    return sorted(list(globals().keys()) + list(_lazy_modules.keys()))

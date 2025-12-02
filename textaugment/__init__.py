import os
import importlib
from types import ModuleType

name = 'textaugment'

__version__ = '2.0.0'
__licence__ = 'MIT'
__author__ = 'Joseph Sefara'
__url__ = 'https://github.com/dsfsi/textaugment/'

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

__all__ = [
    'Translate',
    'Word2vec',
    'Wordnet',
    'EDA',
    'AEDA',
    'MIXUP',
    'LANGUAGES',
]

_lazy_modules: dict[str, tuple[str, str]] = {
    'Translate': ('textaugment.translate', 'Translate'),
    'Word2vec': ('textaugment.word2vec', 'Word2vec'),
    'Wordnet': ('textaugment.wordnet', 'Wordnet'),
    'EDA': ('textaugment.eda', 'EDA'),
    'AEDA': ('textaugment.aeda', 'AEDA'),
    'MIXUP': ('textaugment.mixup', 'MIXUP'),
    'LANGUAGES': ('textaugment.constants', 'LANGUAGES'),
}


def __getattr__(name: str) -> object:
    '''
    Dynamically import modules when attributes are requested.
    '''
    if name in _lazy_modules:
        module_name, attribute_name = _lazy_modules[name]
        module: ModuleType = importlib.import_module(module_name)
        attribute: object = getattr(module, attribute_name)
        globals()[name] = attribute
        return attribute
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    '''Ensure lazy attributes appear in dir().'''
    return sorted(
        list(globals().keys()) + 
        list(_lazy_modules.keys())
    )

import os

from .mbart_translate import MbartTranslate


__version__ = '3.0.0'
__licence__ = 'MIT'
__author__ = 'Isheanesu Joseph Dzingirai'
__url__ = 'https://github.com/dsfsi/textaugment/'

PACKAGE_DIR: str = os.path.dirname(os.path.abspath(__file__))

__all__: list[str] = [
    'MbartTranslate',
]

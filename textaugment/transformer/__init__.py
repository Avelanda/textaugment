import os

from .paraphrase import ParaphraseAugmentor
from .contextual_word import ContextualWordAugmentor

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

__version__ = '3.0.0'
__licence__ = 'MIT'
__author__ = 'Isheanesu Joseph Dzingirai'
__url__ = 'https://github.com/dsfsi/textaugment/'

__all__ = [
    'ParaphraseAugmentor',
    'ContextualWordAugmentor'
]

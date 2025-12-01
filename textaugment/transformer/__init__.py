import os

from .paraphrase import Paraphraser
from .pipeline_util import PipelineHelper
from .contextual_word_substitution import ContextualWordSubstituion
from .controlled_text_generation import ControlTextGeneration
from .style_transfer_domain_adaption import StyleTransferDomainAdapter


__version__ = '3.0.0'
__licence__ = 'MIT'
__author__ = 'Isheanesu Joseph Dzingirai'
__url__ = 'https://github.com/dsfsi/textaugment/'

PACKAGE_DIR: str = os.path.dirname(os.path.abspath(__file__))

__all__: list[str] = [
    'Paraphraser',
    'PipelineHelper',
    'ContextualWordSubstituion',
    'ControlTextGeneration',
    'StyleTransferDomainAdapter'
]

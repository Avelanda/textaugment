import random
from typing import Any

from .pipeline_util import PipelineHelper

import spacy
import spacy.cli
from spacy.language import Language
from spacy.tokens.doc import Doc
from spacy.util import is_package

from transformers import AutoModelForMaskedLM, FillMaskPipeline


class ContextualWordSubstituion:
    '''
    Implements selective word substitution [1] for contextualized  text augmentation.
    It focuses on word modifiers (15%), specifically adjectives (ADJ) and adverbs (ADV). 
    Up to 15% of these tokens in the text are randomly selected for masking. 
    Of the selected tokens, 80% are replaced with the [MASK] token, while the remaining 20% remain unchanged.

    [1] Pantelidou, K., Chatzakou, D., Tsikrika, T., Vrochidis, S., Kompatsiaris, I. (2022).
    Selective Word Substitution for Contextualized Data Augmentation. 
    In: Rosso, P., Basile, V., Martínez, R., Métais, E., Meziane, F. (eds)
    Natural Language Processing and Information Systems. NLDB 2022. 
    Lecture Notes in Computer Science, vol 13286. Springer, Cham. 
    https://doi.org/10.1007/978-3-031-08473-7_47
    '''

    def __init__(
        self, 
        spacy_model:str = 'en_core_web_sm',
        mask_token_probability: float = 0.8,
        max_mask_token_fraction: float = 0.15,
        model_name: str = 'google-bert/bert-base-uncased'
    ) -> None:
        '''
        Initialize the ContextualWordSubstituion object.

        :type spacy_model:                  str
        :param spacy_model:                 spaCy model to load (default: 'en_core_web_sm').
        :type mask_token_probability:       float
        :param mask_token_probability:      Probability of masking a selected word (default: 0.8).
        :type max_mask_token_fraction:      float
        :param max_mask_token_fraction:     Maximum fraction of ADJ/ADV tokens to mask (default: 0.15).
        :type model_name: str        
        :param model_name:                  Masked Language Model (MLM) name (default: 'google-bert/bert-base-uncased').

        :rtype:                             None
        :return:                            None
        '''
        self.__model_name: str = model_name
        self.__mask_token: str | None = None
        self.__pipeline: FillMaskPipeline | None = None
        self.__mask_probability: float = mask_token_probability
        self.__max_mask_token_fraction: float = max_mask_token_fraction

        if not is_package(spacy_model):
            spacy.cli.download(spacy_model)
        self.__nlp: Language = spacy.load(spacy_model)

    
    @property
    def __get_pipeline(self) -> FillMaskPipeline:
        '''
        Lazily loads and returns a FillMaskPipeline

        :rtype:     FillMaskPipeline
        :return:    FillMaskPipeline object for masked word prediction.
        '''
        if self.__pipeline is None:
            self.__pipeline = PipelineHelper.get_pipeline(
                self.__model_name,
                AutoModelForMaskedLM,
                'fill-mask'
            )
            self.__mask_token = self.__pipeline.tokenizer.mask_token
        return self.__pipeline

    def __generate_masked_text(self, text: str) -> str:
        '''
        Masks the text by masking adjectives and adverbs.
        This is done by randomly selecting up to `max_mask_token_fraction` of ADJ/ADV tokens to mask.
        The selected tokens are masked with probability `mask_token_probability`.

        :type text:     str
        :param text:    text to mask.
        :rtype:         str
        :return:        text with some words replaced by the MLM's mask token.
        '''
        doc: Doc = self.__nlp(text)
        word_modifier_indices: list[int] = [
            idx 
            for idx, token in enumerate(doc) 
            if token.pos_ in ['ADJ', 'ADV']
        ]

        if not word_modifier_indices:
            return text

        max_num_of_modifiers_to_mask: int = (
            int(len(word_modifier_indices) * self.__max_mask_token_fraction)
            if word_modifier_indices
            else 0
        )

        num_of_modifiers_to_mask: int = max(1, max_num_of_modifiers_to_mask)
        
        mask_indices: set[int] = (
            set(random.sample(word_modifier_indices, num_of_modifiers_to_mask)) 
            if num_of_modifiers_to_mask > 0
            else set()
        )

        tokens: list[str | None] = [
            self.__mask_token 
            if idx in mask_indices and random.random() < self.__mask_probability 
            else token.text
            for idx, token in enumerate(doc)
        ]

        return ' '.join([token for token in tokens if token])

    def augment(self, text: str) -> str:
        '''
        Contextually augments the text by masking it & replacing  masked tokens with MLM predictions.
        Uses the FillMaskPipeline to replace each masked token with the top predicted word.

        :type text:     str
        :param text:    text to augment
        :rtype:         str
        :return:        contextually augmented text
        '''
        fill_pipeline: FillMaskPipeline = self.__get_pipeline
        masked_text: str = self.__generate_masked_text(text)

        while self.__mask_token in masked_text:
            mask_token_count: int = masked_text.count(self.__mask_token)
            results: list[dict[str, Any]] | list[list[dict[str, Any]]] = fill_pipeline(masked_text)

            top_word_replacement: str = (
                results[0]['token_str'] 
                if mask_token_count == 1 
                else results[0][0]['token_str']
            )
            masked_text = masked_text.replace(self.__mask_token, top_word_replacement, 1)
        
        return masked_text

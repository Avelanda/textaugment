import random

from .pipeline_util import PipelineHelper
from transformers import AutoModelForSeq2SeqLM, Text2TextGenerationPipeline


class Paraphraser:
    def __init__(
        self, 
        num_beams: int = 5,
        max_new_tokens: int | None = 256,
        early_stopping: bool = True,
        num_return_sequences: int = 5,
        model_name: str = 'google-t5/t5-small',
    ) -> None:
        '''
        Initialize the ParaphraseAugmentor object.

        :type num_beams:                int
        :param num_beams:               Number of beams for beam search.
        :type max_new_tokens:           int | None
        :param max_new_tokens:          Maximum number of new tokens to generate.
        :type early_stopping:           bool
        :param early_stopping:          Whether to stop early when all beams finish.
        :type num_return_sequences:     int
        :param num_return_sequences:    Number of paraphrases to return.
        :type model_name:               str
        :param model_name:              Sequence-to-sequence (Seq2Seq) model name (default: `google-t5/t5-small`)
        '''
        self.__model_name: str = model_name
        self.__num_beams: int = num_beams
        self.__max_new_tokens: int | None = max_new_tokens
        self.__early_stopping: bool = early_stopping
        self.__truncation: bool = False if max_new_tokens is None else True
        self.__num_return_sequences: int = num_return_sequences
        self.__pipeline: Text2TextGenerationPipeline | None = None
    
    @property
    def __get_pipeline(self) -> Text2TextGenerationPipeline:
        '''
        Lazily loads and returns a Text2TextGenerationPipeline

        :rtype:     Text2TextGenerationPipeline
        :return:    Text2TextGenerationPipeline object for paraphrasing.
        '''
        if self.__pipeline is None:
            self.__pipeline = PipelineHelper.get_pipeline(
                self.__model_name,
                AutoModelForSeq2SeqLM,
                'text2text-generation'
            )
        return self.__pipeline

    def augment(self, text: str) -> str:
        '''
        Generates `num_return_sequences` paraphrases of the text.
        
        :type text:     str
        :param text:    text to paraphrase
        :rtype:         list[str]
        :return:        The list paraphased texts
        '''
        input_text = (
            f'paraphrase: {text}' 
            if 't5' in self.__model_name.lower() 
            else text
        )

        pipeline_results: list[dict[str, str]] = self.__get_pipeline(
            input_text, 
            num_beams=self.__num_beams,
            truncation=self.__truncation,
            max_new_tokens=self.__max_new_tokens,
            early_stopping=self.__early_stopping,
            num_return_sequences=self.__num_return_sequences
        )

        generated_texts: list[str] = [
            result['generated_text'].lower().replace('paraphrase:', '')
            for result in pipeline_results
        ]

        filtered_generated_texts: list[str] = [
            generated_text
            for generated_text in generated_texts
            if not text in generated_text and len(generated_text) >= 0.8 * len(text)
        ]

        return random.choice(filtered_generated_texts)
        
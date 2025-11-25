import platform

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM, 
    BitsAndBytesConfig,
    Text2TextGenerationPipeline,
    pipeline
)

import torch

from typing import Any, Optional


class ParaphraseAugmentor:
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

        self.__pipeline: Optional[Text2TextGenerationPipeline] = None

        self._set_model_device()

    def _set_model_device(self) -> None:
        '''
        Initialize the torch device for the pipeline.
        MPS on macOS, CUDA if available, else CPU

        :rtype: None
        :return: None
        '''
        if platform.system() == 'Darwin': 
            self.__device: torch.device = torch.device(
                'mps' 
                if torch.backends.mps.is_available() 
                else 'cpu'
            )
        else:
            self.__device: torch.device = torch.device(
                'cuda' 
                if torch.cuda.is_available()
                else 'cpu'
            )
    
    @property
    def __get_pipeline(self) -> Text2TextGenerationPipeline:
        '''
        Lazily loads and returns a Text2TextGenerationPipeline

        :rtype:     Text2TextGenerationPipeline
        :return:    Text2TextGenerationPipeline object for paraphrasing.
        '''
        if self.__pipeline is None:
            tokenizer: Any = AutoTokenizer.from_pretrained(self.__model_name)

            quantization_config: BitsAndBytesConfig | None = (
                BitsAndBytesConfig(load_in_8bit=True)
                if self.__device.type in ['cuda', 'cpu'] 
                else None
            )

            model: Any = AutoModelForSeq2SeqLM.from_pretrained(
                self.__model_name,
                device_map='auto',
                trust_remote_code=True,
                quantization_config=quantization_config,
            )

            model.eval()

            self.__pipeline = pipeline(
                'text2text-generation',
                model=model,
                tokenizer=tokenizer
            )

        return self.__pipeline

    def augment(self, text: str) -> list[str]:
        '''
        Generates `num_return_sequences` paraphrases of the text.
        
        :type text:     str
        :param text:    text to paraphrase
        :rtype:         list[str]
        :return:        The list paraphased texts
        '''
        text = (
            f'paraphrase: {text}' 
            if 't5' in self.__model_name.lower() 
            else text
        )

        
        pipeline_results: list[dict[str, str]] = self.__get_pipeline(
            text, 
            num_beams=self.__num_beams,
            truncation=self.__truncation,
            max_new_tokens=self.__max_new_tokens,
            early_stopping=self.__early_stopping,
            num_return_sequences=self.__num_return_sequences
        )

        return [
            result['generated_text'].lower().replace('paraphrase: ', '')
            for result in pipeline_results
        ]

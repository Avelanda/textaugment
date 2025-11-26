import platform

import random

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    Text2TextGenerationPipeline
)


import torch

from typing import Any, Optional


class StyleTransferDomainAdapter:
    def __init__(
        self,
        num_beams: int = 5,
        max_new_tokens: int | None = 1024,
        early_stopping: bool = True,
        num_return_sequences: int = 5,
        model_name: str = 'google-t5/t5-small'
    ) -> None:
        self.__model_name: str = model_name

        self.__num_beams: int = num_beams
        self.__max_new_tokens: int | None = max_new_tokens
        self.__early_stopping: bool = early_stopping
        self.__truncation: bool = False if max_new_tokens is None else True
        self.__num_return_sequences: int = num_return_sequences

        self.__pipeline: Optional[Text2TextGenerationPipeline] = None


    @property
    def __get_pipeline(self) -> Text2TextGenerationPipeline:
        '''
        Lazily loads and returns a Text2TextGenerationPipeline

        :rtype:     Text2TextGenerationPipeline
        :return:    Text2TextGenerationPipeline object for style transfer domain adaption.
        '''
        if self.__pipeline is None:
            tokenizer: Any = AutoTokenizer.from_pretrained(self.__model_name)

            if platform.system() == 'Darwin': 
                device: str = 'mps' if torch.backends.mps.is_available() else 'cpu'
            else:
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

            quantization_config: BitsAndBytesConfig | None = (
                BitsAndBytesConfig(load_in_8bit=True)
                if device in ['cuda', 'cpu'] 
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


    def augment(
        self, 
        text: str, 
        domain: str='news',
        style: str='informal'
    ) -> list[str]:        
        input_text = f'''
            Rewrite the following text to the {domain} domain with {style} stlye.
            Preserve meaning but change tone and wording.
            Text:
            {text}
        '''
        
        pipeline_results: list[dict[str, str]] = self.__get_pipeline(
            input_text, 
            num_beams=self.__num_beams,
            truncation=self.__truncation,
            max_new_tokens=self.__max_new_tokens,
            early_stopping=self.__early_stopping,
            num_return_sequences=self.__num_return_sequences
        )

        return [
            result['generated_text']
            for result in pipeline_results
            if result['generated_text']
        ]

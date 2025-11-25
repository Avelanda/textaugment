import platform

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM, 
    BitsAndBytesConfig,
    Text2TextGenerationPipeline,
    pipeline
)

import torch

from typing import Any


class Paraphrase:
    def __init__(
        self, 
        model_name: str,
        num_beams: int = 10,
        padding: bool = True,
        max_length: int | None = 256,
        early_stopping: bool = True,
        num_return_sequences: int = 5
    ) -> None:
        '''
        A method to initialize model, tokenizer, and device parameters

        :type model_name:       str
        :param model_name:      Seq2Seq model name
        :rtype:                 None
        :return:                constructer do not return
        '''
        self.__model_name: str = model_name

        self.__padding: bool = padding
        self.__num_beams: int = num_beams
        self.__max_length: int | None = max_length
        self.__early_stopping: bool = early_stopping
        self.__truncation: bool = False if max_length is None else True
        self.__num_return_sequences: int = num_return_sequences

        self.__model = None
        self.__tokenizer = None

        self._set_model_device()

    def _set_model_device(self) -> None:
        '''
        A method to initialize the torch device

        :rtype:     None
        :return:    None
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
    def __get__tokenizer(self) -> Any:
        '''
        Lazily loads the tokenizer on demand

        rtype:  Any
        return: returns the initialized tokenizer 
        '''
        if self.__tokenizer is None:
            self.__tokenizer: Any = AutoTokenizer.from_pretrained(self.__model_name)
        return self.__tokenizer
    
    @property
    def __get_model(self) -> Any:
        '''
        Lazily loads the model

        rtype:  Any
        return: returns the initialized Seq2Seq model
        '''
        if self.__model is None:
            quantization_config: BitsAndBytesConfig | None = (
                BitsAndBytesConfig(load_in_8bit=True)
                if self.__device.type in ['cuda', 'cpu'] 
                else None
            )

            self.__model: Any = AutoModelForSeq2SeqLM.from_pretrained(
                self.__model_name,
                trust_remote_code=True,
                quantization_config=quantization_config,
            )
        

        self.__model.eval()
        return self.__model

    def augment(self, text: str) -> list[str]:
        '''
        A method to paraphrase the text using a Seq2Seq model.
        
        :type text:     str
        :param text:    text used for paraphrasing
        :rtype:         list[str]
        :return:        The list of possible paraphases (num_return_sequences) of the text
        '''
        tokenizer: Any = self.__get__tokenizer
        model: Any = self.__get_model

        text = (
            f'paraphrase: {text}' 
            if 't5' in self.__model_name.lower() 
            else text
        )

        paraphrase_pipeline: Text2TextGenerationPipeline = pipeline(
            'text2text-generation',
            model=model,
            tokenizer=tokenizer
        )

        pipeline_results: list[dict[str, str]] = paraphrase_pipeline(
            text, 
            num_beams=self.__num_beams,
            truncation=self.__truncation,
            max_new_tokens=self.__max_length,
            early_stopping=self.__early_stopping,
            num_return_sequences=self.__num_return_sequences
        )

        return [
            result['generated_text'].lower().replace('paraphrase: ', '')
            for result in pipeline_results
        ]

        


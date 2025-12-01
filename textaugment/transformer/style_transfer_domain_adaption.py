import random

from .pipeline_util import PipelineHelper
from transformers import AutoModelForSeq2SeqLM, Text2TextGenerationPipeline


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

        self.__pipeline: Text2TextGenerationPipeline | None = None


    @property
    def __get_pipeline(self) -> Text2TextGenerationPipeline:
        '''
        Lazily loads and returns a Text2TextGenerationPipeline

        :rtype:     Text2TextGenerationPipeline
        :return:    Text2TextGenerationPipeline object for style transfer domain adaption.
        '''
        if self.__pipeline is None:
            self.__pipeline = PipelineHelper.get_pipeline(
                self.__model_name,
                AutoModelForSeq2SeqLM,
                'text2text-generation'
            )
        return self.__pipeline

    def augment(
        self, 
        text: str, 
        domain: str='news',
        style: str='informal'
    ) -> str:        
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

        generated_texts: list[str] = [
            result['generated_text']
            for result in pipeline_results
            if result['generated_text']
        ]
    
        filtered_generated_texts: list[str] = [
            generated_text
            for generated_text in generated_texts
            if not text in generated_text and len(generated_text) >= 0.8 * len(text)
        ]

        return random.choice(filtered_generated_texts)
        

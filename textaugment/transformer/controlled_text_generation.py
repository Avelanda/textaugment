from dataclasses import dataclass, field
from dotenv import load_dotenv

from enum import Enum

from .pipeline_util import PipelineHelper

import textwrap

from transformers import AutoModelForCausalLM, Pipeline


class Sentiment(Enum):
    POSITIVE = 'positive'
    NEGATIVE = 'negative'
    NEUTRAL = 'neutral'


class Style(Enum):
    FORMAL = 'formal'
    INFORMAL = 'informal'


class ReadingLevel(Enum):
    BASIC = 'basic'
    INTERMEDIATE = 'intermediate'
    ADVANCED = 'advanced'


class Length(Enum):
    SHORT = 'short'
    MEDIUM = 'medium'
    LONG = 'long'


@dataclass
class TextGenerationControls:
    sentiment: Sentiment = Sentiment.POSITIVE
    style: Style = Style.FORMAL
    reading_level: ReadingLevel = ReadingLevel.ADVANCED
    length: Length = Length.SHORT
    include_keywords: list[str] = field(default_factory=list)
    exclude_keywords: list[str] = field(default_factory=list)


class ControlTextGeneration:
    def __init__(
        self,
        max_new_tokens: int = 50,
        model_name: str = 'tiiuae/falcon-7b-instruct',
    ) -> None:
        self.__model_name: str = model_name
        self.__max_new_tokens: int = max_new_tokens
        self.__pipeline: Pipeline | None = None

        load_dotenv()

    @property
    def __get_pipeline(self) -> Pipeline:
        '''
        Lazily loads and returns a TextGenerationPipeline

        :rtype:     TextGenerationPipeline
        :return:    TextGenerationPipeline object for controlled text generation.
        '''
        if self.__pipeline is None:
            self.__pipeline = PipelineHelper.get_pipeline(
                self.__model_name,
                AutoModelForCausalLM,
                'text-generation'
            )
        return self.__pipeline

    def __generate_prompt(
        self, 
        text: str, 
        controls: TextGenerationControls
    ) -> str:
        include_keywords: str = ', '.join(controls.include_keywords) if controls.include_keywords else ''
        exclude_keywords: str = ', '.join(controls.exclude_keywords) if controls.exclude_keywords else ''

        return textwrap.dedent(f'''
        You are a friendly chatbot.
        Rewrite the following text using the specified controls.

        Controls:
        - Sentiment: {controls.sentiment.value}
        - Style: {controls.style.value}
        - Reading Level: {controls.reading_level.value}
        - Length: {controls.length.value}
        - Include keywords: {include_keywords}
        - Exclude keywords: {exclude_keywords}

        User: {text}
        Assistant:
        ''').lstrip()
    
    def augment(self, text: str, controls: TextGenerationControls | None = None) -> str:
        if controls is None:
            controls = TextGenerationControls()

        prompt: str = self.__generate_prompt(text, controls)

        generated_texts: list[dict[str, str]] = self.__get_pipeline(
            prompt, 
            max_new_tokens=self.__max_new_tokens, 
            do_sample=False
        )

        generated_text: str = generated_texts[0]['generated_text']
        new_text: str = generated_text[len(prompt):].strip()
        return new_text.split('\n')[0]

from typing import Any

import torch

from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    logging,
    pipeline
)


class PipelineHelper:
    @staticmethod
    def __get_model(
        model_name: str, 
        model_loader_class: Any
    ) -> PreTrainedModel:
        gpu_available: bool = torch.cuda.is_available()

        if gpu_available:
            return model_loader_class.from_pretrained(
                model_name,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                low_cpu_mem_usage=True,
                device_map='auto'
            ).eval()
        else:
            return model_loader_class.from_pretrained(
                model_name,
                device_map='auto'
            ).eval()

    @staticmethod
    def get_pipeline(
        model_name: str,
        model_loader_class: Any,
        pipeline_task: str
    ) -> Any:
        logging.set_verbosity_error()
        
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
        model: PreTrainedModel = PipelineHelper.__get_model(model_name, model_loader_class)

        return pipeline(
            task=pipeline_task,
            model=model,
            tokenizer=tokenizer,
            use_fast=True
        )

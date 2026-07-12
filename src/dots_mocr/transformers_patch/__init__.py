from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

from .configuration_dots_ocr import DotsOCRConfig, DotsVisionConfig
from .modeling_dots_ocr import DotsOCRForCausalLM
from .processing_dots_ocr import DotsVLProcessor


def register_transformers():
    AutoConfig.register("dots_ocr", DotsOCRConfig, exist_ok=True)
    AutoModelForCausalLM.register(DotsOCRConfig, DotsOCRForCausalLM, exist_ok=True)
    AutoProcessor.register(DotsOCRConfig, DotsVLProcessor, exist_ok=True)


__all__ = [
    "DotsOCRConfig",
    "DotsOCRForCausalLM",
    "DotsVLProcessor",
    "DotsVisionConfig",
    "register_transformers",
]

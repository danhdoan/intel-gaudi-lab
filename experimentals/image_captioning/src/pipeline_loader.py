"""Load image captioning pipeline with Gaudi support."""

import torch
from habana_frameworks.torch.hpu import wrap_in_hpu_graph
from optimum.habana.transformers.modeling_utils import (
    adapt_transformers_to_gaudi,
)
from src.registry import IMAGE_CAPTIONING_MODEL
from transformers import AutoConfig, AutoProcessor, pipeline

# ==============================================================================


def load_image_captioning_pipeline(model_path) -> pipeline:
    """Load the image captioning pipeline with Gaudi support.

    Args:
    ----
        model_path (str): Path to the model.

    Returns:
    -------
        pipeline: The image captioning pipeline.
        prompt (str): The prompt for the model.

    """
    adapt_transformers_to_gaudi()
    torch._C._set_math_sdp_allow_fp16_bf16_reduction(True)
    model_path = IMAGE_CAPTIONING_MODEL[0]
    config = AutoConfig.from_pretrained(model_path)
    model_dtype = torch.bfloat16
    model_type = config.model_type

    if model_type in [
        "llava",
        "idefics2",
        "llava_next",
        "mllama",
        "paligemma",
        "qwen2_vl",
    ]:
        processor = AutoProcessor.from_pretrained(
            model_path, padding_side="left"
        )
        prompt = ""
        if processor.chat_template:
            conv = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is shown in this image?",
                        },
                        {"type": "image"},
                    ],
                }
            ]
            prompt = processor.apply_chat_template(
                conv, add_generation_prompt=True
            )
        else:
            img_tok = getattr(config, "image_token_id", None) or getattr(
                config, "image_token_index", None
            )
            img_str = (
                processor.tokenizer.convert_ids_to_tokens(img_tok)
                if img_tok is not None
                else "<image>"
            )
            prompt = f"{img_str}\nWhat is shown in this image?"

    pipe = pipeline(
        "image-to-text",
        model=model_path,
        tokenizer=model_path,
        image_processor=model_path,
        torch_dtype=model_dtype,
        device="hpu",
    )

    pipe.model = wrap_in_hpu_graph(pipe.model)
    if model_type in [
        "idefics2",
        "mllama",
        "paligemma",
        "qwen2_vl",
        "llava",
        "llava_next",
    ]:
        from transformers.image_utils import load_image

        def preprocess(self, image, prompt=None, timeout=None):
            kwargs = {"max_length": 500, "padding": "max_length"}
            image = load_image(image, timeout=timeout)
            model_inputs = processor(
                images=image,
                text=prompt,
                return_tensors=self.framework,
                **kwargs,
            )
            return model_inputs

        pipe.__class__.preprocess = preprocess

    return pipe, prompt


# ==============================================================================

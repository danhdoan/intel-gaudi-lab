"""Inference Application

with Stable Diffusion XL Base 1.0 Model
"""

__author__ = ["Nicolas Howard", "Nguyen Tran"]
__email__ = ["petit.nicolashoward@gmail.com", "nguyen.tran@enouvo.com"]
__date__ = "2025/04/04"
__status__ = "development"


# ======================================================================================


import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from accelerate import PartialState
from compel import Compel, ReturnedEmbeddingsType

from optimum.habana.diffusers import (
    GaudiDDIMScheduler,
    GaudiEulerAncestralDiscreteScheduler,
    GaudiEulerDiscreteScheduler,
    GaudiFlowMatchEulerDiscreteScheduler,
)
from optimum.habana.utils import set_seed

from experiments.get_sd_args import get_args


# ======================================================================================


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)


# ======================================================================================


def get_scheduler(args)
    if args.scheduler == "euler_discrete":
        scheduler = GaudiEulerDiscreteScheduler.from_pretrained(
            args.model_name_or_path, subfolder="scheduler", **kwargs
        )
        if args.optimize:
            scheduler.hpu_opt = True
    elif args.scheduler == "euler_ancestral_discrete":
        scheduler = GaudiEulerAncestralDiscreteScheduler.from_pretrained(
            args.model_name_or_path, subfolder="scheduler", **kwargs
        )
    elif args.scheduler == "ddim":
        scheduler = GaudiDDIMScheduler.from_pretrained(args.model_name_or_path, subfolder="scheduler", **kwargs)
    else:
        scheduler = None
        
    return scheduler


# ======================================================================================


def get_prompts(args):
    pass


# ======================================================================================


def main():
    """Main Application."""

    args = get_args()

    if args.optimize and not args.use_habana:
        raise ValueError("--optimize can only be used with --use-habana.")

    # Select stable diffuson pipeline based on input
    sdxl_models = ["stable-diffusion-xl", "sdxl"]
    sdxl = True if any(model in args.model_name_or_path for model in sdxl_models) else False


    # Set the scheduler
    kwargs = {"timestep_spacing": args.timestep_spacing, "rescale_betas_zero_snr": args.use_zero_snr}

    scheduler = get_scheduler(args)



    # Set pipeline class instantiation options
    kwargs = {
        "use_habana": args.use_habana,
        "use_hpu_graphs": args.use_hpu_graphs,
        "gaudi_config": args.gaudi_config_name,
        "sdp_on_bf16": args.sdp_on_bf16,
    }

    if scheduler is not None:
        kwargs["scheduler"] = scheduler

    if args.bf16:
        kwargs["torch_dtype"] = torch.bfloat16

    # Set pipeline call options
    kwargs_call = {
        "num_images_per_prompt": args.num_images_per_prompt,
        "batch_size": args.batch_size,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "eta": args.eta,
        "output_type": args.output_type,
        "profiling_warmup_steps": args.profiling_warmup_steps,
        "profiling_steps": args.profiling_steps,
    }

    if args.width > 0 and args.height > 0:
        kwargs_call["width"] = args.width
        kwargs_call["height"] = args.height

    if args.use_cpu_rng:
        kwargs_call["generator"] = torch.Generator(device="cpu").manual_seed(args.seed)
    else:
        kwargs_call["generator"] = None


    # ! LoRA
    if args.lora_scale is not None:
        kwargs_call["lora_scale"] = args.lora_scale

    negative_prompts = args.negative_prompts
    if args.distributed:
        distributed_state = PartialState()
        if args.negative_prompts is not None:
            with distributed_state.split_between_processes(args.negative_prompts) as negative_prompt:
                negative_prompts = negative_prompt
    kwargs_call["negative_prompt"] = negative_prompts

    if sdxl:
        prompts_2 = args.prompts_2
        if args.distributed and args.prompts_2 is not None:
            with distributed_state.split_between_processes(args.prompts_2) as prompt_2:
                prompts_2 = prompt_2
        kwargs_call["prompt_2"] = prompts_2

    if sdxl:
        negative_prompts_2 = args.negative_prompts_2
        if args.distributed and args.negative_prompts_2 is not None:
            with distributed_state.split_between_processes(args.negative_prompts_2) as negative_prompt_2:
                negative_prompts_2 = negative_prompt_2
        kwargs_call["negative_prompt_2"] = negative_prompts_2


    kwargs_call["quant_mode"] = args.quant_mode

    # Instantiate a Stable Diffusion pipeline class
    quant_config_path = os.getenv("QUANT_CONFIG")


    if args.optimize:
        # Import SDXL pipeline
        # set PATCH_SDPA to enable fp8 varient of softmax in sdpa
        os.environ["PATCH_SDPA"] = "1"
        import habana_frameworks.torch.hpu as torch_hpu

        from optimum.habana.diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_mlperf import (
            StableDiffusionXLPipeline_HPU,
        )

        pipeline = StableDiffusionXLPipeline_HPU.from_pretrained(
            args.model_name_or_path,
            **kwargs,
        )

        pipeline.unet.set_default_attn_processor(pipeline.unet)
        pipeline.to(torch.device("hpu"))

        if quant_config_path:
            import habana_frameworks.torch.core as htcore
            from neural_compressor.torch.quantization import FP8Config, convert, prepare

            htcore.hpu_set_env()

            config = FP8Config.from_json_file(quant_config_path)

            if config.measure:
                logger.info("Running measurements")
                pipeline.unet = prepare(pipeline.unet, config)
            elif config.quantize:
                logger.info("Running quantization")
                pipeline.unet = convert(pipeline.unet, config)
            htcore.hpu_initialize(pipeline.unet, mark_only_scales_as_const=True)

        if args.use_hpu_graphs:
            pipeline.unet = torch_hpu.wrap_in_hpu_graph(pipeline.unet)

    else:
        from optimum.habana.diffusers import GaudiStableDiffusionXLPipeline

        pipeline = GaudiStableDiffusionXLPipeline.from_pretrained(
            args.model_name_or_path,
            **kwargs,
        )

    # Load LoRA weights if provided
    if args.lora_id:
        pipeline.load_lora_weights(args.lora_id)


    # Set RNG seed
    set_seed(args.seed)
    if args.use_compel:
        tokenizer = [pipeline.tokenizer]
        text_encoder = [pipeline.text_encoder]
        if sdxl:
            tokenizer.append(pipeline.tokenizer_2)
            text_encoder.append(pipeline.text_encoder_2)
            compel = Compel(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                device=torch.device("cpu"),
            )
        else:
            compel = Compel(tokenizer=tokenizer, text_encoder=text_encoder, device=torch.device("cpu"))

    if args.use_freeu:
        if args.use_hpu_graphs:
            raise ValueError("Freeu cannot support the HPU graph model, please disable it.")

        pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.5, b2=1.6)

    # If prompts file is specified override prompts from the file
    if args.prompts_file is not None:
        lines = []
        with open(args.prompts_file, "r") as file:
            lines = file.readlines()
        lines = [line.strip() for line in lines]
        args.prompts = lines


def generate_outputs(args)
    # Generate Images using a Stable Diffusion pipeline
    if args.distributed:
        with distributed_state.split_between_processes(args.prompts) as prompt:
            if args.use_compel:
                if sdxl:
                    conditioning, pooled = compel(prompt)
                    outputs = pipeline(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, **kwargs_call)
                else:
                    prompt_embeds = compel(prompt)
                    outputs = pipeline(prompt_embeds=prompt_embeds, **kwargs_call)
            else:
                outputs = pipeline(prompt=prompt, **kwargs_call)
    else:
        if args.use_compel:
            if sdxl:
                conditioning, pooled = compel(args.prompts)
                outputs = pipeline(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, **kwargs_call)
            else:
                prompt_embeds = compel(args.prompts)
                outputs = pipeline(prompt_embeds=prompt_embeds, **kwargs_call)
        else:
            outputs = pipeline(prompt=args.prompts, **kwargs_call)

    if args.optimize and quant_config_path and config.measure:
        from neural_compressor.torch.quantization import finalize_calibration

        logger.info("Finalizing calibration...")
        finalize_calibration(pipeline.unet)
    
    save_images(*args, outputs)


def save_images(args, outputs):
    # Save images in the specified directory if not None and if they are in PIL format
    if args.image_save_dir is not None:
        if args.output_type == "pil":
            image_save_dir = Path(args.image_save_dir)
            if args.distributed:
                image_save_dir = Path(f"{image_save_dir}_{distributed_state.process_index}")

            image_save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving images in {image_save_dir.resolve()}...")
        
            for i, image in enumerate(outputs.images):
                image.save(image_save_dir / f"image_{i + 1}.png")
        else:
            logger.warning("--output_type should be equal to 'pil' to save images in --image_save_dir.")


# ======================================================================================


if __name__ == "__main__":
    main()


# ======================================================================================

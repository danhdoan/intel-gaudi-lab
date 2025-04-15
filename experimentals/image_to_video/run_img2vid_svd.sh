#!/bin/bash

python3 experimentals/image_to_video/image_to_video_generation_with_svd.py \
  --model_path "models/stabilityai/stable-video-diffusion-img2vid-xt" \
  --use_habana \
  --use_hpu_graphs \
  --gaudi_config_name Habana/stable-diffusion \
  --sdp_on_bf16 \
  --bf16 \
  --num_videos_per_prompt 1 \
  --batch_size 1 \
  --height 576 \
  --width 1024 \
  --num_inference_steps 25 \
  --min_guidance_scale 1.0 \
  --max_guidance_scale 3.0 \
  --fps 7 \
  --motion_bucket_id 127 \
  --noise_aug_strength 0.02 \
  --decode_chunk_size 25 \
  --output_type "pil" \
  --profiling_warmup_steps 0 \
  --profiling_steps 0 \
  --seed 42 \
  --image_path "data/rocket" \
  --output_path "outputs"

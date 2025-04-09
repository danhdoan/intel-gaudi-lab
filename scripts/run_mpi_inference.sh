python experimentals/distributed_runner/gaudi_spawn.py \
    --world_size 4 \
    --use_mpi \
    experimentals/sd_xl_base_1/sd_xl_base_1_inference.py \
    --model_name_or_path "models/stable-diffusion-xl-base-1.0" \
    --prompts "An image of falling leaves in an avenue in Monet style" \
    --negative_prompts "tedious" \
    --num_images_per_prompt 10 \
    --batch_size 8 \
    --num_inference_steps 30 \
    --image_save_dir output/mpi/$(date +'%y%m%d_%H%M') \
    --scheduler ddim \
    --height 512 \
    --width 512 \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16 \
    --distributed

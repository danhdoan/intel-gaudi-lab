
from optimum.habana import GaudiConfig
from optimum.habana.diffusers import GaudiDDIMScheduler, GaudiStableDiffusionPipeline
from datetime import datetime

model_name = "stabilityai/stable-diffusion-xl-base-1.0"
output_dir ="outputs"
scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

pipeline = GaudiStableDiffusionPipeline.from_pretrained(
    model_name,
    scheduler=scheduler,
    use_habana=True,
    use_hpu_graphs=True,
    gaudi_config="Habana/stable-diffusion",
)

outputs = pipeline(
    ["Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"],
    num_images_per_prompt=16,
    batch_size=4,
)
output_image = outputs.images[0]
for i, image in enumerate(outputs.images):
    image.save(f"{output_dir}/{timestamp}_{i}.png")
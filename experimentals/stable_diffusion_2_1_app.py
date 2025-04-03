from fastapi import FastAPI
import uvicorn
import torch
from optimum.habana import GaudiConfig
from optimum.habana.diffusers import GaudiDDIMScheduler, GaudiStableDiffusionPipeline
from optimum.habana.utils import set_seed

import base64
from io import BytesIO

#====================================================================
app = FastAPI()

# Initialize pipeline globally
pipe = None


#====================================================================
@app.on_event("startup")
async def startup_event():
    """
    Load the model and scheduler on startup.
    This function is called when the FastAPI application starts.
    """
    global pipe
    model_name = "models/stable-diffusion-2.1"
    scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
    pipe = GaudiStableDiffusionPipeline.from_pretrained(
        model_name,
        scheduler=scheduler,
        use_habana=True,
        use_hpu_graphs=True,
        gaudi_config="Habana/stable-diffusion-2",
    )
    pipe(prompt="test", num_inference_steps=30).images[0]
    pipe(prompt="test", num_inference_steps=30).images[0]

#====================================================================
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

#====================================================================
@app.post("/generate")
async def generate_response(
    prompt: str,
    negative_prompt: str | None = None,
    num_inference_steps: int = 30,
    width: int = 512,
    height: int = 512,
    batch_size: int = 1,
    guidance_scale: float = 7.5,
    seed: int = 42,
):
    """
    Generate an image based on the provided prompt.
    Args:
        prompt (str): The prompt to generate an image from.
        negative_prompt (str | None): Optional negative prompt to guide the generation.
    Returns:
        dict: A dictionary containing the generated image in base64 format.
    """
    global pipe

    # Set seed
    set_seed(seed)
    # Set the generator for reproducibility

    image = (
        pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            batch_size=batch_size,
            guidance_scale=guidance_scale,
        ).images[0],
    )
    buffered = BytesIO()
    image.save(buffered, format="PNG")

    img_str = base64.b64encode(buffered.getvalue()).decode()
    return {"message": "Image generated successfully", "image": img_str}

#====================================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

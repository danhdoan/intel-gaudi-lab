"""FastAPI application for generating images using Stable Diffusion 2.1."""

import base64
import os
from io import BytesIO

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from optimum.habana import utils as habana_utils
from optimum.habana.diffusers import (
    GaudiDDIMScheduler,
    GaudiStableDiffusionPipeline,
)
from pydantic import BaseModel

# ====================================================================


app = FastAPI()

app.state.pipe = None


# ====================================================================


class GenerateRequest(BaseModel):
    """Request model for image generation.

    Attributes:
        prompt (str): The text prompt to generate the image.
        negative_prompt (str | None): The negative prompt
        num_inference_steps (int): The number of inference steps to take.
        width (int): The width of the generated image.
        height (int): The height of the generated image.
        batch_size (int): The number of images to generate in a batch.
        guidance_scale (float): The scale for classifier-free guidance.
        seed (int): The random seed for reproducibility.

    """

    prompt: str
    negative_prompt: str | None = None
    num_inference_steps: int = 30
    width: int = 512
    height: int = 512
    batch_size: int = 1
    guidance_scale: float = 7.5
    seed: int = 42


# ====================================================================


class GenerateResponse(BaseModel):
    """Response model for image generation.

    Attributes:
        image (str): The generated image in Base64 format.

    """

    image: str  # Base64-encoded image data


# ====================================================================


@app.on_event("startup")
async def startup_event():
    """Load the model and scheduler on startup.

    This function is called when the FastAPI application starts.
    """
    model_name = "models/stable-diffusion-2.1"
    scheduler = GaudiDDIMScheduler.from_pretrained(
        model_name, subfolder="scheduler"
    )
    app.state.pipe = GaudiStableDiffusionPipeline.from_pretrained(
        model_name,
        scheduler=scheduler,
        use_habana=True,
        use_hpu_graphs=True,
        gaudi_config="Habana/stable-diffusion-2",
    )
    app.state.pipe(
        prompt="a beautiful landscape with mountains and a river",
        num_inference_steps=30,
    ).images[0]
    app.state.pipe(
        prompt="a magical forest with glowing mushrooms",
        num_inference_steps=30,
    ).images[0]


# ====================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint to verify the service is running.

    Returns:
        dict: A dictionary indicating the service status.

    """
    return {"status": "healthy"}


# ====================================================================


@app.post("/generate", response_model=GenerateResponse)
async def generate_response(
    request: GenerateRequest,
):
    """Generate an image based on the provided prompt.

    Args:
    ----
        request (GenerateRequest): The request object


    Returns:
    -------
        GenerateResponse: The response object containing the generated image.

    """
    # Set seed
    habana_utils.set_seed(request.seed)
    # Set the generator for reproducibility
    image = app.state.pipe(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        num_inference_steps=request.num_inference_steps,
        height=request.height,
        width=request.width,
        batch_size=request.batch_size,
        guidance_scale=request.guidance_scale,
    ).images[0]

    buffered = BytesIO()
    image.save(buffered, format="PNG")

    img_str = base64.b64encode(buffered.getvalue()).decode()
    return GenerateResponse(image=img_str)


# ====================================================================


app.mount("/", StaticFiles(directory="public", html=True), name="static")


@app.get("/")
async def root():
    """Serve the index.html file."""
    with open(os.path.join("public", "index.html")) as file:
        return HTMLResponse(content=file.read())


# ====================================================================


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

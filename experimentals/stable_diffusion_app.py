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
    GaudiStableDiffusion3Pipeline,
    GaudiStableDiffusionPipeline,
    GaudiStableDiffusionXLPipeline,
)
from pydantic import BaseModel

# ====================================================================


app = FastAPI()

app.state.pipe = None


# ====================================================================


class GenerateRequest(BaseModel):
    """Request model for image generation.

    Args:
    ----
        prompt (str): The text prompt to generate the image.
        negative_prompt (str | None): The negative prompt
        num_inference_steps (int): The number of inference steps to take.
        width (int): The width of the generated image.
        height (int): The height of the generated image.
        num_images_per_prompt (int): The number of images to generate
        batch_size (int): The number of images to generate in a batch.
        guidance_scale (float): The scale for classifier-free guidance.
        seed (int): The random seed for reproducibility.

    """

    prompt: str
    negative_prompt: str | None = None
    num_inference_steps: int = 30
    width: int = 512
    height: int = 512
    num_images_per_prompt: int = 1
    batch_size: int = 1
    guidance_scale: float = 7.5
    seed: int = 42


# ====================================================================


class GenerateResponse(BaseModel):
    """Response model for image generation.

    Args:
    ----
        image (str): The generated list of base64 encoded images.

    """

    image: list[str]


# ====================================================================


@app.on_event("startup")
async def startup_event():
    """Load the model and scheduler on startup.

    This function is called when the FastAPI application starts.
    """
    model_name = "./models/stable-diffusion-2.1"
    scheduler = GaudiDDIMScheduler.from_pretrained(
        model_name, subfolder="scheduler"
    )

    pipeline_mapping = {
        "stable-diffusion-xl": GaudiStableDiffusionXLPipeline,
        "stable-diffusion-3": GaudiStableDiffusion3Pipeline,
    }
    pipeline_class = next(
        (cls for key, cls in pipeline_mapping.items() if key in model_name),
        GaudiStableDiffusionPipeline,
    )
    app.state.pipe = pipeline_class.from_pretrained(
        model_name,
        scheduler=scheduler,
        use_habana=True,
        use_hpu_graphs=True,
        gaudi_config="Habana/stable-diffusion",
    )


# ====================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint to verify the service is running.

    Returns
    -------
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
    # start_time = time.time()
    habana_utils.set_seed(request.seed)
    list_base64 = []
    outputs = app.state.pipe(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        num_inference_steps=request.num_inference_steps,
        height=request.height,
        width=request.width,
        num_images_per_prompt=request.num_images_per_prompt,
        batch_size=request.batch_size,
        guidance_scale=request.guidance_scale,
    )
    images = outputs["images"]
    # calculate time taken
    # time_taken = time.time() - start_time
    for i in range(len(images)):
        buffered = BytesIO()
        images[i].save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        images[i] = img_str
        list_base64.append(img_str)

    return GenerateResponse(image=list_base64)


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

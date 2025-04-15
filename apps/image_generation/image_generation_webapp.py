"""FastAPI Web App for Image Generation using Gaudi Inference Pipeline."""

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from src.process_image import image_to_base64
from src.type import GenerateRequest, GenerateResponse

from libs.generative_ai.image_generation.stable_diffusion.gaudi_inference_pipeline import (  # noqa
    GaudiStableDiffusionInferencePipeline,
)  # noqa: E501

# ====================================================================


app = FastAPI()


# ====================================================================


@app.on_event("startup")
async def startup_event():
    """Startup event to initialize the Gaudi SD Inference Pipeline."""
    app.state.pipeline = GaudiStableDiffusionInferencePipeline()


# ====================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint to verify the service is running."""
    return {"status": "healthy"}


# =====================================================================


@app.post("/change_model")
async def change_model(model_name: str) -> dict:
    """Endpoint to change the model dynamically.

    Args:
    ----
        model_name (str): The name of the new model.
        The model name should be one of the allowed models.
        Allowed models are:
        - stable-diffusion-3-m-d
        - stable-diffusion-2.1
        - stable-diffusion-2-base
        - stable-diffusion-xl-base-1.0
        The model should be located in the "./models" directory.

    Returns:
    -------
        dict: A dictionary indicating the status of the operation.

    """
    allowed_models = [
        "stable-diffusion-3-m-d",
        "stable-diffusion-2.1",
        "stable-diffusion-2-base",
        "stable-diffusion-xl-base-1.0",
    ]
    if model_name not in allowed_models:
        return {"status": "error", "message": "Invalid model name"}

    app.state.pipeline.change_model(model_name=model_name)

    return {"status": "success", "message": f"Model changed to {model_name}"}


# ====================================================================


@app.post("/generate", response_model=GenerateResponse)
async def generate_response(
    request: GenerateRequest,
) -> GenerateResponse:
    """Generate an image based on the provided prompt.

    Args:
    ----
        request (GenerateRequest): The request object containing the parameters
        for image generation.
        prompt (str): The text prompt to generate the image.
        negative_prompt (str | None): The negative prompt to guide generation.
        num_inference_steps (int): The number of inference steps to take.
        width (int): The width of the generated image.
        height (int): The height of the generated image.
        num_images_per_prompt (int): The number of images to generate.
        batch_size (int): The number of images to generate in a batch.
        guidance_scale (float): The scale for classifier-free guidance.
        seed (int): The random seed for reproducibility.

    Returns:
    -------
        GenerateResponse: The response object containing the generated image.

    """
    images = app.state.pipeline.generate_image(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        num_inference_steps=request.num_inference_steps,
        height=request.height,
        width=request.width,
        num_images_per_prompt=request.num_images_per_prompt,
        batch_size=request.batch_size,
        guidance_scale=request.guidance_scale,
        seed=request.seed,
    )
    base64_images = image_to_base64(images)
    return GenerateResponse(image=base64_images)


# ====================================================================


app.mount("/", StaticFiles(directory="public", html=True), name="static")


# ====================================================================


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)

# ====================================================================

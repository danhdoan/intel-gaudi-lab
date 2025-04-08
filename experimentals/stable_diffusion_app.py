"""FastAPI application for generating images using Stable Diffusion 2.1."""

import base64
import os
from io import BytesIO

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from macro import DIFFUSION_MODEL, MODEL_PATH_FOLDER
from optimum.habana import utils as habana_utils
from stable_diffusion.pipeline_loader import load_pipeline
from type import GenerateRequest, GenerateResponse

# ====================================================================


app = FastAPI()

app.state.pipe = None


# ====================================================================


@app.on_event("startup")
async def startup_event() -> None:
    """Load the default model and scheduler on startup.

    This function is called when the FastAPI application starts.
    It loads the Stable Diffusion model from the specified
    directory and initializes the pipeline.

    """
    model_name = f"{MODEL_PATH_FOLDER}/{DIFFUSION_MODEL[0]}"
    app.state.pipe = load_pipeline(model_name)


# ====================================================================


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
    if model_name not in DIFFUSION_MODEL:
        return {
            "status": "error",
            "message": f"Model name {model_name} is not allowed. ",
        }

    try:
        if app.state.pipe is not None:
            del app.state.pipe
        model_name = f"{MODEL_PATH_FOLDER}/{model_name}"

        app.state.pipe = load_pipeline(model_name)
        return {
            "status": "success",
            "message": f"Model changed to {model_name}",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


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
) -> GenerateResponse:
    """Generate an image based on the provided prompt.

    Args:
    ----
        request (GenerateRequest): The request object


    Returns:
    -------
        GenerateResponse: The response object containing the generated image.

    """
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

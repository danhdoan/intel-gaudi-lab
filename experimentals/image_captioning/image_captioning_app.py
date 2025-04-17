"""FastAPI application for image captioning using a pre-trained model."""

import re
from io import BytesIO

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image
from src.pipeline_loader import load_image_captioning_pipeline
from src.registry import IMAGE_CAPTIONING_MODEL
from src.type import GenerateResponse

# ==============================================================================


app = FastAPI()
app.state.pipe = None
app.state.model_path = IMAGE_CAPTIONING_MODEL[0]
app.state.prompt = ""


# ==============================================================================


@app.on_event("startup")
async def startup_event() -> None:
    """Load the image captioning pipeline on startup."""
    app.state.pipe, app.state.prompt = load_image_captioning_pipeline(
        app.state.model_path
    )


# ==============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint.

    Returns
    -------
        dict: Health status.

    """
    return {"status": "healthy"}


# ==============================================================================


@app.post("/generate", response_model=GenerateResponse)
async def generate_caption(image: UploadFile = File(...)) -> GenerateResponse:
    """Generate a caption for the uploaded image.

    Args:
    ----
        image (UploadFile): The uploaded image file.

    Returns:
    -------
        GenerateResponse: The generated caption.

    """
    if app.state.pipe is None:
        return {"status": "error", "message": "Pipeline not loaded."}
    raw = await image.read()
    img = Image.open(BytesIO(raw)).convert("RGB")
    prompt = app.state.prompt
    output = app.state.pipe(
        images=img,
        prompt=prompt,
        max_new_tokens=500,
    )
    raw_text = output[0].get("generated_text")
    parts = re.split(r"assistant\s*:?\s*", raw_text, flags=re.IGNORECASE)
    caption = parts[-1].strip()
    return GenerateResponse(answer=caption)


# ==============================================================================


@app.post("/change_model")
async def change_model(model_path: str):
    """Change the model used for image captioning.

    Args:
    ----
        model_path (str): The path of the new model.
        The model path should be one of the allowed models.
        Allowed models are:
        - /intel-gaudi-lab/models/llava-hf/llava-v1.6-vicuna-13b-hf
        - /intel-gaudi-lab/models/meta-llama/Llama-3.2-11B-Vision-Instruct
    Returns:
        dict: Status message.

    """
    if not model_path:
        return {"status": "error", "message": "Model path cannot be empty."}

    if model_path not in IMAGE_CAPTIONING_MODEL:
        return {
            "status": "error",
            "message": f"Invalid model path: {model_path}.",
        }

    if model_path == app.state.model_path:
        return {"status": "success", "model": model_path}

    try:
        del app.state.pipe
        del app.state.prompt
        app.state.pipe, app.state.prompt = load_image_captioning_pipeline(
            model_path
        )
        app.state.model_path = model_path
        return {"status": "success", "model": model_path}
    except Exception as e:
        return {"status": "error", "message": f"Failed to load model: {str(e)}"}


# ==============================================================================


app.mount("/", StaticFiles(directory="public", html=True), name="static")


# ==============================================================================


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)


# ==============================================================================

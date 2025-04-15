"""FastAPI application for image-to-video generation using Stable Diffusion."""

import json
from io import BytesIO

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from optimum.habana import utils as habana_utils
from PIL import Image
from src.macro import IMAGE_TO_VIDEO_MODEL, MODEL_PATH_FOLDER
from src.pipeline_loader import load_pipeline
from src.process_video import process_video
from src.type import GenerateRequest, GenerateResponse

# ====================================================================


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.pipe = None


# ====================================================================


@app.on_event("startup")
async def startup_event() -> None:
    """Load the default model and scheduler on startup.

    This function is called when the FastAPI application starts.
    It loads the Stable Diffusion model from the specified
    directory and initializes the pipeline.

    """
    model_name = f"{MODEL_PATH_FOLDER}/{IMAGE_TO_VIDEO_MODEL}"
    app.state.pipe = load_pipeline(model_name)


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
    request_data: str = Form(...),
    image: UploadFile = File(...),
) -> GenerateResponse:
    """Generate a video from an image using the Stable Diffusion model.

    Args:
    ----
        request_data (str): JSON string containing the generation parameters.
        image (UploadFile): The input image file.

    Returns:
    -------
        GenerateResponse: A response object containing the generated video.

    """
    request = GenerateRequest(**json.loads(request_data))

    habana_utils.set_seed(request.seed)
    raw_image = await image.read()
    image_decoded = Image.open(BytesIO(raw_image)).convert("RGB")
    output_video = app.state.pipe(
        prompt=request.prompt,
        image=image_decoded,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        negative_prompt=request.negative_prompt,
        num_frames=request.nums_frames,
        num_videos_per_prompt=request.num_video_per_prompt,
    )
    video_bytes = []
    for i, frames in enumerate(output_video.frames):
        video_bytes.append(
            process_video(
                frames,
                fps=request.fps,
            )
        )

    return GenerateResponse(image=video_bytes)


# ====================================================================


app.mount("/", StaticFiles(directory="public", html=True), name="static")


# ====================================================================


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)

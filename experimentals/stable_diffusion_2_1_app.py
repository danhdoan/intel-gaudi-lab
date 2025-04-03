from fastapi import FastAPI
import uvicorn
import torch
from optimum.habana import GaudiConfig
from optimum.habana.diffusers import GaudiDDIMScheduler, GaudiStableDiffusionPipeline
from optimum.habana.utils import set_seed
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

import base64
from io import BytesIO

#====================================================================
app = FastAPI()

# Initialize pipeline globally
pipe = None
class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    num_inference_steps: int
    width: int
    height: int
    batch_size: int
    guidance_scale: float
    seed: int

class GenerateResponse(BaseModel):
    image: str  # URL or Base64-encoded image data

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
    # pipe(prompt="test", num_inference_steps=30).images[0]
    # pipe(prompt="test", num_inference_steps=30).images[0]

#====================================================================
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

#====================================================================
@app.post("/generate",response_model=GenerateResponse)
async def generate_response(
    request:GenerateRequest,
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
    set_seed(request.seed)
    # Set the generator for reproducibility
    image = pipe(
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
    # return {"message":"..."}

app.mount("/", StaticFiles(directory="public", html=True), name="static")
@app.get("/")
async def root():
    """
    Serve the index.html file.
    """
    with open(os.path.join("public", "index.html"), "r") as file:
        return HTMLResponse(content=file.read())

#====================================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

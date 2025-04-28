"""Text generation server implementation using FastAPI and Gaudi accelerators.

This module provides a web server for text generation using LLM,
optimized for Intel Gaudi HPUs. It includes model loading, warm-up procedures,
and a RESTful API for text generation.
"""

import logging
import os
import time

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from optimum.habana.transformers.modeling_utils import (
    adapt_transformers_to_gaudi,
)
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Configure rank of process
RANK = int(os.environ.get("RANK", 0))
PORT = 8000 + RANK
os.environ["HABANA_VISIBLE_DEVICES"] = str(RANK)
device = torch.device(f"hpu:{RANK}")

adapt_transformers_to_gaudi()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

MODEL_PATH = os.environ.get("MODEL_PATH")

class TextGenerator:
    """Handles text generation using a language model optimized for HPUs."""

    def __init__(self, model_path):
        """Initialize the text generator.

        Args:
            model_path: Path to the pre-trained language model

        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.initialized = False

    def initialize(self):
        """Load and prepare the model and tokenizer for text generation.

        This method loads the model and tokenizer, configures padding tokens,
        moves the model to the HPU device, and performs warm-up.
        """
        if self.initialized:
            return

        logger.info(f"Initializing model from {self.model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Set padding side
        if not self.model.config.is_encoder_decoder:
            self.tokenizer.padding_side = "left"

        # Ensure padding token is properly set
        if self.model.generation_config.pad_token_id is None:
            if isinstance(self.model.generation_config.eos_token_id, int):
                self.model.generation_config.pad_token_id = (
                    self.model.generation_config.eos_token_id
                )
            elif isinstance(self.model.generation_config.eos_token_id, list):
                self.model.generation_config.pad_token_id = (
                    self.model.generation_config.eos_token_id[0]
                )
        self.tokenizer.bos_token_id = self.model.generation_config.bos_token_id
        if isinstance(self.model.generation_config.eos_token_id, int):
            self.tokenizer.eos_token_id = self.model.generation_config.eos_token_id
        elif isinstance(self.generation_config.eos_token_id, list):
            self.tokenizer.eos_token_id = self.model.generation_config.eos_token_id[0]
        self.tokenizer.pad_token_id = self.model.generation_config.pad_token_id
        self.tokenizer.pad_token = self.tokenizer.decode(self.tokenizer.pad_token_id)
        self.tokenizer.eos_token = self.tokenizer.decode(self.tokenizer.eos_token_id)
        self.tokenizer.bos_token = self.tokenizer.decode(self.tokenizer.bos_token_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.generation_config.pad_token_id = (
                self.model.generation_config.eos_token_id
            )

        self.model = self.model.eval().to("hpu")
        self.model = torch.compile(
            self.model, backend="hpu_backend", options={"keep_input_mutations": True})

        from habana_frameworks.torch.hpu import wrap_in_hpu_graph

        self.model = wrap_in_hpu_graph(self.model)

        # Warm up the model
        self.warm_up()

        self.initialized = True
        logger.info("Model initialization completed")

    def warm_up(self, num_iterations=3):
        """Warm up the model by running inference multiple times.

        Args:
            num_iterations: Number of warm-up iterations to perform

        """
        logger.info("Starting model warm-up...")
        warm_up_prompt = "This is a warm-up run to optimize the model."

        for i in range(num_iterations):
            try:
                logger.info(f"Warm-up iteration {i+1}/{num_iterations}")
                # Tokenize input
                input_tokens = self.tokenizer(
                    warm_up_prompt, return_tensors="pt", padding=True
                )

                # Move inputs to device
                for t in input_tokens:
                    if torch.is_tensor(input_tokens[t]):
                        input_tokens[t] = input_tokens[t].to("hpu")

                # Generate text
                with torch.no_grad():  # Use no_grad for inference/warmup
                    # Run model generation but don't use outputs in warm-up
                    self.model.generate(
                        **input_tokens,
                        max_new_tokens=10,
                        use_cache=True,
                        lazy_mode=True,
                        hpu_graphs=True,
                        trim_logits=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    ).cpu()

                # Synchronize to ensure completion
                torch.hpu.synchronize()
            except Exception as e:
                logger.error(
                    f"Error during warm-up iteration {i+1}: {e}", exc_info=True
                )
                break  # Stop warm-up on error

        logger.info("Model warm-up completed")

    def generate(self, prompt, max_new_tokens=500):
        """Generate text based on the provided prompt.

        Args:
            prompt: The input text to base generation on
            max_new_tokens: Maximum number of new tokens to generate

        Returns:
            Generated text or error message

        """
        if not self.initialized:
            logger.warning("Generator called before initialization.")
            try:
                self.initialize()
            except Exception as init_e:
                logger.error(
                    f"Failed to initialize generator dynamically: {init_e}",
                    exc_info=True,
                )
                return "Error: Generator not initialized."

        try:
            # Start time tracking
            start_time = time.time()

            # Tokenize input
            input_tokens = self.tokenizer(
                prompt, return_tensors="pt", padding=True
            )

            # Move inputs to device
            for t in input_tokens:
                if torch.is_tensor(input_tokens[t]):
                    input_tokens[t] = input_tokens[t].to("hpu")

            # Generate text using no_grad context for better performance
            
            outputs=[]
            for token in self.generate_from_model(max_new_tokens, input_tokens):
                outputs.append(token)
                print(token)
            # Decode generated text
            generated_text_list = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

            # Extra safety check
            if not generated_text_list or not isinstance(
                generated_text_list[0], str
            ):
                logger.warning(
                    f"Unexpected decode result: {type(generated_text_list)}"
                )
                return "No text was generated."

            # Log generation time
            end_time = time.time()
            logger.info(
                f"Text generated in {end_time - start_time:.2f} seconds"
            )

            # Return raw text without any formatting
            return generated_text_list[0]

        except Exception as e:
            logger.error(f"Error during generation: {str(e)}", exc_info=True)
            return f"Error during generation: {str(e)}"

    def generate_from_model(self,max_new_tokens, input_tokens):
        with torch.no_grad():
            for token in self.model.generate(
                **input_tokens,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                lazy_mode=True,
                hpu_graphs=True,
                trim_logits=True,
                pad_token_id=self.tokenizer.pad_token_id,
            ).cpu():
                yield token
            # Synchronize to ensure completion
            torch.hpu.synchronize()

# Initialize FastAPI app
app = FastAPI()

# Mount static files using absolute path
app.mount("/static", StaticFiles(directory=TEMPLATES_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Initialize the generator
generator = TextGenerator(MODEL_PATH)


class GenerationRequest(BaseModel):
    """Request model for text generation API endpoint."""

    prompt: str
    max_new_tokens: int = 500


@app.on_event("startup")
async def startup_event():
    """Initialize the text generator when the server starts."""
    # Initialize the generator when the server starts
    try:
        generator.initialize()
    except Exception as e:
        logger.critical(
            f"Failed to initialize text generator on startup: {e}",
            exc_info=True,
        )


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main page of the application."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate")
async def generate_text(request: GenerationRequest):
    """Generate text based on the provided prompt.

    Args:
        request: The generation request containing prompt and parameters

    Returns:
        JSON response with the generated text or error message

    """
    try:
        # Add timing for API request handling
        start_time = time.time()

        # Generate text
        output = generator.generate(request.prompt, request.max_new_tokens)

        # Log total API request time
        end_time = time.time()
        logger.info(
            f"Total API request handled in {end_time - start_time:.2f} seconds"
        )

        # Check if error occurred
        if output.startswith("Error:"):
            return {"status": "error", "message": output}
        else:
            return {"status": "success", "output": output}
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)

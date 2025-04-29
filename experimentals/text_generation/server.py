"""Text to Text Generation Server.

This module provides a web server using FastAPI and Gaudi accelerators for LLM
text generation, optimized for Intel Gaudi HPUs.
"""

__author__ = ["Cuong Do", "Nicolas Howard"]
__email__ = ["vanlanhdh@gmail.com", "petit.nicolashoward@gmail.com"]
__date__ = "2025/04/28"
__status__ = "development"


# ==============================================================================


import json
import logging
import os
import time

import habana_frameworks.torch.core as htcore
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from optimum.habana.transformers.modeling_utils import (
    adapt_transformers_to_gaudi,
)
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==============================================================================


# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ==============================================================================


# Configure rank of process
RANK = int(os.environ.get("RANK", 0))
PORT = 8000 + RANK
os.environ["HABANA_VISIBLE_DEVICES"] = str(RANK)

# Configure model and device
MODEL_PATH = os.environ.get("MODEL_PATH")

DEVICE = torch.device("hpu")
adapt_transformers_to_gaudi()

# Configure web server resources
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")


# ==============================================================================


class GenerationRequest(BaseModel):
    """Request model for text generation API endpoint."""

    prompt: str
    max_new_tokens: int = 100
    streaming: bool = False


class GenerationResponse(BaseModel):
    """Response model for text generation API endpoint."""

    text: str
    tokens: list[str] | None = None
    generation_time: float


# ==============================================================================


class TextGenerator:
    """Handles text generation using a language model optimized for HPUs."""

    def __init__(self, model_path):
        """Initialize the text generator."""
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.initialized = False

    def initialize(self):
        """Load and prepare the model and tokenizer for text generation."""
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

        # Configure tokenizer
        self.tokenizer.bos_token_id = self.model.generation_config.bos_token_id
        self.tokenizer.eos_token_id = (
            self.model.generation_config.eos_token_id
            if isinstance(self.model.generation_config.eos_token_id, int)
            else self.model.generation_config.eos_token_id[0]
        )
        self.tokenizer.pad_token_id = self.model.generation_config.pad_token_id

        # Set special tokens
        self.tokenizer.pad_token = self.tokenizer.decode(
            self.tokenizer.pad_token_id
        )
        self.tokenizer.eos_token = self.tokenizer.decode(
            self.tokenizer.eos_token_id
        )
        self.tokenizer.bos_token = self.tokenizer.decode(
            self.tokenizer.bos_token_id
        )

        # Fallback for pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.generation_config.pad_token_id = (
                self.model.generation_config.eos_token_id
            )

        # Prepare model
        self.model = self.model.eval().to(DEVICE)
        self.model = torch.compile(
            self.model,
            backend="hpu_backend",
            options={"keep_input_mutations": True},
        )

        # Wrap model in HPU graph
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph
        self.model = wrap_in_hpu_graph(self.model)

        # Warm up the model
        self.warm_up()

        self.initialized = True
        logger.info("Model initialization completed")

    def warm_up(self, num_iterations=3):
        """Warm up the model by running inference multiple times."""
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
                        input_tokens[t] = input_tokens[t].to(DEVICE)

                # Generate text
                with torch.no_grad():
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
                break

        logger.info("Model warm-up completed")

    def generate(self, prompt, max_new_tokens=500):
        """Generate text based on the provided prompt."""
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
            start_time = time.time()

            # Tokenize input
            input_tokens = self.tokenizer(
                prompt, return_tensors="pt", padding=True
            )

            # Move inputs to device
            for t in input_tokens:
                if torch.is_tensor(input_tokens[t]):
                    input_tokens[t] = input_tokens[t].to(DEVICE)

            # Generate text
            outputs = self.model.generate(
                **input_tokens,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                lazy_mode=True,
                hpu_graphs=True,
                trim_logits=True,
                pad_token_id=self.tokenizer.pad_token_id,
            ).cpu()

            # Synchronize to ensure completion
            torch.hpu.synchronize()

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

            return generated_text_list[0]

        except Exception as e:
            logger.error(f"Error during generation: {str(e)}", exc_info=True)
            return f"Error during generation: {str(e)}"


# ==============================================================================


# Initialize FastAPI app
app = FastAPI()

# Mount static files using absolute path
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Initialize the generator
generator = TextGenerator(MODEL_PATH)


# ==============================================================================


@app.on_event("startup")
async def startup_event():
    """Initialize the text generator when the server starts."""
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
    """Generate text based on the input prompt."""
    try:
        start_time = time.time()
        
        # Tokenize input
        input_tokens = generator.tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(DEVICE)

        if request.streaming:
            async def generate_stream():
                # Initialize generation state
                current_text = ""
                current_tokens = []
                
                # Generate tokens one at a time
                for _ in range(request.max_new_tokens):
                    # Generate next token
                    outputs = generator.model.generate(
                        **input_tokens,
                        max_new_tokens=1,
                        use_cache=True,
                        lazy_mode=True,
                        hpu_graphs=True,
                        trim_logits=True,
                        pad_token_id=generator.tokenizer.pad_token_id,
                    ).cpu()
                    
                    # Extract and decode new token
                    new_token = outputs[0][-1].item()
                    current_tokens.append(new_token)
                    token_text = generator.tokenizer.decode(
                        [new_token], 
                        skip_special_tokens=True,
                    )
                    
                    # Update current text
                    current_text += token_text
                    
                    # Update input context for next iteration
                    input_tokens["input_ids"] = torch.cat([
                        input_tokens["input_ids"],
                        torch.tensor([[new_token]], device=DEVICE),
                    ], dim=1)
                    input_tokens["attention_mask"] = torch.cat([
                        input_tokens["attention_mask"],
                        torch.tensor([[1]], device=DEVICE),
                    ], dim=1)
                    
                    # Prepare and send current state
                    data = {
                        'token': token_text,
                        'text': current_text,
                    }
                    sse_message = f"data: {json.dumps(data)}\n\n"
                    yield sse_message
                    
                    # Mark graph step
                    htcore.mark_step()
                
                # Prepare and send completion message
                generation_time = time.time() - start_time
                final_data = {
                    'done': True,
                    'generation_time': generation_time,
                }
                final_message = f"data: {json.dumps(final_data)}\n\n"
                yield final_message

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
            )
        
        else:
            # Standard non-streaming generation
            outputs = generator.model.generate(
                **input_tokens,
                max_new_tokens=request.max_new_tokens,
                use_cache=True,
                lazy_mode=True,
                hpu_graphs=True,
                trim_logits=True,
                pad_token_id=generator.tokenizer.pad_token_id,
            ).cpu()
            
            # Decode full text
            generated_text = generator.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            )
            
            # Return complete response
            generation_time = time.time() - start_time
            return GenerationResponse(
                text=generated_text,
                generation_time=generation_time,
            )

    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during generation: {str(e)}",
        )


# ==============================================================================


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)


# ==============================================================================

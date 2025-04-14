# Image to Video Generation App

This application uses Stable Diffusion to generate videos from input images.

## Running the Application

The application can be run using the provided `run_app.sh` script. This script executes the `image_to_video_generation_app.py` file, which starts the FastAPI server.

1.  Make the `run_app.sh` script executable:

    ```bash
    chmod +x run_app.sh
    ```

2.  Run the application:

    ```bash
    ./run_app.sh
    ```

    This will start the FastAPI server, by default on `0.0.0.0:8008`.

## Interacting with the Application

The application provides a web interface (served from the `public` directory) and a `/generate` endpoint for creating videos.

### Web Interface

The web interface allows you to upload an image and specify generation parameters.  It is served at the root `/` endpoint.

### API Endpoint

The `/generate` endpoint accepts POST requests with the following:

-   `request_data`: A JSON string containing the generation parameters.  See `src/type.py` for the `GenerateRequest` data model.
-   `image`: The image file to use for video generation.

Example using `curl`:

```bash
curl -X POST -F 'request_data={"prompt": "a cat", "seed": 42, "num_inference_steps": 50, "guidance_scale": 7.5, "negative_prompt": "blurry", "nums_frames": 16, "nums_video_per_prompt": 1, "fps": 24}' -F 'image=@/path/to/your/image.jpg' http://localhost:8008/generate > output.json
```

The response will be a JSON object containing the generated video data.  See `src/type.py` for the `GenerateResponse` data model.

## Configuration

The application uses the following environment variables (defined in `src/macro.py`):

-   `IMAGE_TO_VIDEO_MODEL`: The name of the Stable Diffusion model to use.
-   `MODEL_PATH_FOLDER`: The directory where the model is stored.

These can be modified directly in `src/macro.py` or set as environment variables.

## Health Check

A health check endpoint is available at `/health`.  It returns a JSON response with the status of the service.

```bash
curl http://localhost:8008/health
```

## Notes

-   Ensure that the specified model exists in the `MODEL_PATH_FOLDER`.
-   The `public` directory contains the static files for the web interface.
-   The `src` directory contains the application logic, including the pipeline loading, video processing, and data models.

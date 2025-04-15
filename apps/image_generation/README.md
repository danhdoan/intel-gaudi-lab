# Documentation: Running the Stable Diffusion Application

## Overview
The `run_app.sh` script is a simple Bash script designed to execute the `image_generation_webapp` FastAPI application. This application provides endpoints for generating images using the Stable Diffusion 2.1 model, dynamically switching models, and performing health checks.

## Running the Application
To start the application, follow these steps:

1. **Navigate to the Script Directory**:
    Open a terminal and navigate to the directory containing the `run_app.sh` script:
    ```bash
    cd apps/image_generation

    ```

2. **Make the Script Executable** (if not already):
    Ensure the `run_app.sh` script has executable permissions:
    ```bash
    chmod +x run_app.sh
    ```

3. **Run the Script**:
    Execute the script to start the FastAPI application:
    ```bash
    ./run_app.sh
    ```

4. **Access the Application**:
    Once the application is running, it will be accessible at `http://0.0.0.0:8002` by default. You can use a web browser or tools like `curl` to interact with the API.

## Key Endpoints
- **Health Check**: Verify the service is running:
  ```
  GET /health
  ```
- **Generate Image**: Generate an image based on a prompt:
  ```
  POST /generate
  ```
- **Change Model**: Dynamically switch the model:
  ```
  POST /change_model
  ```

## Notes
- The application serves a static HTML file (`index.html`) from the `public` directory at the root endpoint (`/`).
- Ensure the Habana utilities and pipeline loader are correctly configured for optimal performance on Habana hardware.

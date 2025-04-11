# intel-gaudi-lab
Experiments and optimizations with Intel Gaudi for AI training and inference.

## I. Introduction
This project explores training and deployment of AI models on Intel Gaudi hardware with *habana* framework.

## II. Features
- **Image to Video Generation** <br/>
    🗓️ *Released:* April 11, 2025 <br/>
    🔗 *Link:* [image_to_video](experimentals/image_to_video/run_img2vid.sh) <br/>
    <img src="assets/img2v_demo.gif" alt="Image to Video Demo" width="400"/> <br/>
    <sub><i>Prompt: ""</i></sub>

- **Image Generation** <br/>
    🗓️ *Released:* April 08, 2025 <br/>
    🔗 *Link:* [image_generation](experimentals/image_generation/README.md) <br/>
    <img src="assets/img_gen_demo.png" alt="Image Generation Demo" width="400"/> <br/>
    <sub><i>Prompt: "Sailing ship floating gently in the sunflower field painting by Van Gogh"</i></sub>

## III. Installation
1. (Recommended) Prepare Docker Container for development:
    - Duplicate `example.env` to `.env` and edit the configurations
        - `DOCKER_CONTAINER` - container name
        - `WORKING_DIR` - working directory inside the container
    - Run following command to create the container for `habana_frameworks` service:

        ```bash
        make up
        ```

    - Run following command to connect to the booted container:

        ```bash
        make connect
        ```

    - When exited, run following command to stop and remove the created container:

        ```bash
        make down
        ```

2. Prepare Python virtual environment for development:
    - Run following command to activate the virtual environment for this project:

        ```bash
        bash scripts/source_env.sh
        ```

    - In case above command is failed (eg. current setup is not be able to install defined dependencies in `requirements.txt`), you may manually activate the environment and reinstall packages by running the following commands:

        ```bash
        source .venv/bin/activate
        uv add -r requirements.txt
        ```

2. (Optional) Install `pre-commit`, which will run on every commit to identify simple issues before submission to code review:

    ```bash
    pre-commit install
    ```

    ```bash
    pre-commit run --all-files
    ```

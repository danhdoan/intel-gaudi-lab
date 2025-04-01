# intel-gaudi-lab
Experiments and optimizations with Intel Gaudi for AI training and inference.

## I. Introduction
- TBD

## II. Prerequisites
1. Prepare Python virtual environment for developing:
    - Run following command to activate the virtual environment for this project:

        ```bash
        bash scripts/source_env.sh
        ```

    - In case above command is failed (eg. current setup is not be able to install defined dependencies in `requirements.txt`), you may manually activate the environment and reinstall packages by running the following commands:

        ```bash
        source .venv/bin/activate
        pip install -r requirements.txt
        ```

2. Install `pre-commit`, which will run on every commit to identify simple issues before submission to code review:

    ```bash
    pre-commit install

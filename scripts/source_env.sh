#!/bin/bash


# ==============================================================================


VENV_NAME=".venv"
REQUIREMENT_FILE="requirements.txt"


# ==============================================================================


if [ ! -d "${VENV_NAME}" ]
then
	echo "Virtual environment is setting up..."

  # Install `uv` if not available
  if ! command -v uv &> /dev/null; then
      echo "Installing uv..."
      curl -LsSf https://astral.sh/uv/install.sh | sh
  fi

  # Install Python 3.10 using `uv` (only if not already installed)
  if ! uv python list | grep -q "3.10"; then
      uv python install 3.10
  fi
  uv venv "${VENV_NAME}" --python 3.10

  uv add -r "${REQUIREMENT_FILE}"
	echo "Virtual environment setup done! - Name:" "${VENV_NAME}"
fi

source "${VENV_NAME}"/bin/activate

echo "Virtual environment activated!"


# ==============================================================================

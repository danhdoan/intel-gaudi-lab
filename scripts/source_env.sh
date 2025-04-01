#!/bin/bash


# ==============================================================================


VENV_NAME=.venv
REQUIREMENT_FILE=requirements.txt


# ==============================================================================


if [ ! -d ${VENV_NAME} ]
then
	echo "Virtual environment is setting up..."

  # install `uv`
  curl -LsSf https://astral.sh/uv/install.sh | sh
  uv python install 3.10 # use 3.10 as default Python version
  uv venv ${VENV_NAME} --python 3.10

  uv add -r ${REQUIREMENT_FILE}
	echo "Virtual environment setup done! - Name:" ${VENV_NAME}
fi

source ${VENV_NAME}/bin/activate

echo "Virtual environment activated!"


# ==============================================================================

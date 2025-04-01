#!/bin/bash


# ==============================================================================


VENV_NAME=.venv
PYTHON_PATH=`which python3`
REQUIREMENT_FILE=requirements.txt


# ==============================================================================


if [ ! -d ${VENV_NAME} ]
then
	echo "Virtual environment is setting up..."

	${PYTHON_PATH} -m venv ${VENV_NAME}
	source ${VENV_NAME}/bin/activate
	pip install -r ${REQUIREMENT_FILE}

	echo "Virtual environment setup done! - Name:" ${VENV_NAME}
else
	source ${VENV_NAME}/bin/activate
fi

echo "Virtual environment activated!"


# ==============================================================================

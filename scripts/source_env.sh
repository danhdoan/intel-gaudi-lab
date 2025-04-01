#!/bin/bash


# ==============================================================================


HABANALABS_VIRTUAL_DIR=.venv
REQUIREMENT_FILE=requirements.txt
GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so

# ==============================================================================


if [ ! -d ${HABANALABS_VIRTUAL_DIR} ]
then
	echo "Virtual environment is setting up..."

  # Install Habanalabs Environment
  wget -nv https://vault.habana.ai/artifactory/gaudi-installer/1.20.1/habanalabs-installer.sh
  chmod +x habanalabs-installer.sh

  ./habanalabs-installer.sh install --type base --venv

  source ${HABANALABS_VIRTUAL_DIR}/bin/activate

  pip install -r ${REQUIREMENT_FILE}
	echo "Virtual environment setup done! - Name:" ${HABANALABS_VIRTUAL_DIR}
fi

source ${HABANALABS_VIRTUAL_DIR}/bin/activate

echo "Virtual environment activated!"


# ==============================================================================

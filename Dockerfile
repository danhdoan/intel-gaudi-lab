FROM vault.habana.ai/gaudi-docker/1.20.1/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest

ARG WORKING_DIR="intel-gaudi-lab"

WORKDIR /${WORKING_DIR}

COPY . /${WORKING_DIR}/

# RUN pip install --no-cache-dir .
RUN echo "Hehe"

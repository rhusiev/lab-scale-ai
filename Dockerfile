FROM --platform=amd64 nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# Install updates and OS deps
RUN apt update && apt install -y \
    git \
    software-properties-common \
    curl \
    vim

# Install and setup Python and pip
RUN add-apt-repository -y ppa:deadsnakes/ppa \
    && apt install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Set python3.11 to default python3 and python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Copy the current directory contents into the image
COPY . /app

# Set the working directory
WORKDIR /

# Install all requirements
RUN pip install -r /app/requirements.txt

CMD ["/bin/bash", "/app/tasks/fact-checking.sh"]

EXPOSE 80
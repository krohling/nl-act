FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

RUN apt-get update --allow-unauthenticated && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libgl1 \
    libglib2.0-0 \
    wget \
    xvfb \
    unzip \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Install git lfs for large file support
RUN git lfs install

# Setup conda environment
RUN conda init bash && \
    conda create -n nl-act python=3.8.10 -y && \
    echo "source activate nl-act" >> ~/.bashrc

# Install dependencies
WORKDIR /opt/ml/nl-act
COPY requirements.txt /opt/ml/nl-act/requirements.txt
RUN pip install -r requirements.txt

ENV HF_HOME /opt/ml/input/data/huggingface_cache

COPY . /opt/ml/nl-act/

ENTRYPOINT ["./run.sh"]

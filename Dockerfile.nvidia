FROM nvidia/cuda:9.0-cudnn7-devel
# we need -devel instead of -runtime because we are compiling pytorch ourself, so it will run on old GPUs

RUN apt-get update
RUN apt-get install -y wget bzip2 ca-certificates
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH="/opt/conda/bin:${PATH}"
WORKDIR /opt/openautoml/runner

RUN pip install --upgrade pip

RUN pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.10.1-cp36-cp36m-linux_x86_64.whl

RUN apt-get install -y git

ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# Install basic dependencies
RUN conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
RUN conda install -c mingfeima mkldnn

# Add LAPACK support for the GPU
RUN conda install -c pytorch magma-cuda90

RUN apt-get install -y build-essential

RUN git clone --branch v0.4.1 --depth 1 https://github.com/pytorch/pytorch.git
RUN cd pytorch && git submodule update --init
RUN cd pytorch && python setup.py install
RUN rm -rf pytorch

COPY requirements.txt .
RUN pip install -r requirements.txt


COPY src/ src/
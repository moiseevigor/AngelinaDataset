FROM pytorch/pytorch

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    git

# Set the working directory
WORKDIR /app

RUN pip install tensorboardX matplotlib 

RUN python -c "import torchvision; torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)"
RUN python -c "import torchvision; torchvision.models.detection.retinanet_resnet50_fpn_v2(pretrained=True)"
RUN python -c "import torchvision; torchvision.models.resnet50(pretrained=True)"
RUN pip install torch-lr-finder

# Copy the code
# COPY requirements.txt /app/

# RUN mkdir -p /app/src
# RUN pip install -r /app/requirements.txt

# RUN apt install -y liblouis*
# RUN pip install pytorch-ignite tensorboardX python-Levenshtein

# # Fetch build dependencies
# RUN apt-get update && apt-get install -y \
#     autoconf \
#     automake \
#     curl \
#     libtool \
#     libyaml-dev \
#     make \
#     pkg-config \
#     python \
#     texinfo \
#    && rm -rf /var/lib/apt/lists/*

# # compile and install liblouis
# ADD liblouis /usr/src/liblouis
# WORKDIR /usr/src/liblouis
# RUN ./autogen.sh && ./configure --enable-ucs4 && make && make install && ldconfig

# # install python bindings
# WORKDIR /usr/src/liblouis/python
# RUN python setup.py install

# # clean up
# WORKDIR /app
# RUN rm -rf /usr/src/liblouis
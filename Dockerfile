# Copyright 2017 Diamond Light Source
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

FROM quay.io/rosalindfranklininstitute/jax:v0.3.1

# Install dependencies
RUN apt-get install -y ffmpeg libsm6 libxext6 && \
    apt-get autoremove -y --purge && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Build specific OpenMPI with extensions
RUN mkdir -p /tmp/openmpi && \
    wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.0.tar.gz -P /tmp/openmpi && \
    tar xf /tmp/openmpi/openmpi-4.0.0.tar.gz -C /tmp/openmpi
WORKDIR /tmp/openmpi/openmpi-4.0.0/build
RUN ../configure --prefix=/usr/local \
                 --enable-mpi1-compatibility \
                 --disable-mpi-fortran && \
    make && \
    make install && \
    rm -rf /tmp/openmpi && \
    ldconfig
WORKDIR /

# Install python packages
RUN env MPICC=/usr/local/bin/mpicc pip install --no-cache-dir --upgrade mpi4py && \
    rm -rf /tmp/* && \
    find /usr/lib/python3.*/ -name 'tests' -exec rm -rf '{}' +

# Build radia (only need radia.so on the the PYTHONPATH)
RUN mkdir -p /tmp/radia && \
    git clone https://github.com/ochubar/Radia.git /tmp/radia && \
    make -C /tmp/radia/cpp/gcc all && \
    make -C /tmp/radia/cpp/py && \
    mkdir -p /usr/local/radia && \
    cp /tmp/radia/env/radia_python/radia.so /usr/local/radia/radia.so  && \
    rm -rf /tmp/*
ENV PYTHONPATH="/usr/local/radia:${PYTHONPATH}"

# Install Opt-ID dependencies to enable caching
ADD requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt && \
    rm -rf /tmp/* && \
    find /usr/lib/python3.*/ -name 'tests' -exec rm -rf '{}' +

# Install Opt-ID
ADD . /usr/local/optid
WORKDIR /usr/local/optid
RUN pip install -e . && \
    rm -rf /tmp/* && \
    find /usr/lib/python3.*/ -name 'tests' -exec rm -rf '{}' +

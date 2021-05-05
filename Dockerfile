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

FROM ubuntu:20.04

# Install packages
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && \
    apt-get update -y && apt-get install -y software-properties-common && \
    add-apt-repository universe && \
    apt-get update -y && apt-get install -y dialog apt-utils && \
    apt-get install -y build-essential git curl python2 python2-dev ffmpeg libsm6 libxext6 libopenmpi-dev openmpi-bin && \
    apt-get autoremove -y --purge && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install PIP2
RUN curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py && \
    python2 get-pip.py && \
    rm -f get-pip.py

# Install python packages
RUN pip2 install --no-cache-dir --upgrade \
        mock pytest pytest-cov PyYAML coverage \
        more_itertools numpy h5py scipy matplotlib && \
    env MPICC=/usr/local/bin/mpicc pip2 install --no-cache-dir --upgrade \
        mpi4py && \
    rm -rf /tmp/* && \
    find /usr/lib/python2.*/ -name 'tests' -exec rm -rf '{}' +

# Install Opt-ID
ADD . /usr/local/Opt-ID
WORKDIR /usr/local/Opt-ID
RUN pip2 install -e . && \
    rm -rf /tmp/* && \
    find /usr/lib/python2.*/ -name 'tests' -exec rm -rf '{}' +
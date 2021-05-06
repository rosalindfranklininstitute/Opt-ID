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
    apt-get update -y && apt-get install -y dialog apt-utils && \
    apt-get install -y build-essential git python3-pip ffmpeg libsm6 libxext6 libopenmpi-dev openmpi-bin && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10 && \
    apt-get autoremove -y --purge && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install python packages
RUN pip3 install --no-cache-dir --upgrade \
        mock pytest pytest-cov PyYAML coverage \
        more_itertools numpy h5py scipy matplotlib \
        ruamel.yaml Jinja2 pandoc jupyter jupyter_client nbformat nbconvert && \
    env MPICC=/usr/local/bin/mpicc pip3 install --no-cache-dir --upgrade \
        mpi4py && \
    rm -rf /tmp/* && \
    find /usr/lib/python3.*/ -name 'tests' -exec rm -rf '{}' +

# Install Opt-ID
ADD . /usr/local/Opt-ID
WORKDIR /usr/local/Opt-ID
RUN pip3 install -e . && \
    rm -rf /tmp/* && \
    find /usr/lib/python3.*/ -name 'tests' -exec rm -rf '{}' +
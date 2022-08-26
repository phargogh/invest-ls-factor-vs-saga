FROM debian:bullseye

RUN apt update \
        && apt install -y saga \
            wget \
            gdal-bin \
            python3-gdal \
            python3-pip \
            build-essential \
            python3-dev \
            python3-numpy \
            python3-scipy \
            python3-shapely \
            python3-rtree \
            python3-pandas \
            cython3 \
            python3 \
            git \
            python3-venv \
        && apt-get clean

RUN python3 -m pip install \
            build \
            setuptools_scm \
            babel \
            taskgraph \
            pint \
            pygeoprocessing \
        && python3 -m pip cache purge

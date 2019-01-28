FROM nvidia/cuda:10.0-devel-ubuntu18.04

RUN apt-get update -q -y && apt-get install -y -q zlib1g-dev
ADD https://cmake.org/files/v3.13/cmake-3.13.2-Linux-x86_64.sh /cmake-3.13.2-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.13.2-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

COPY . /app
WORKDIR /app/build
CMD cmake .. && make && cuda-memcheck ./Allen -v 100 /input/minbias/banks/ -d /input/minbias/MC_info/ -g /input/detector_configuration/ -f /input/minbias/banks/ -i /input/minbias/true_pvs/

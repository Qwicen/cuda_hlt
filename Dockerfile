FROM nvidia/cuda:9.2-devel-ubuntu18.04

RUN apt-get update -y
RUN apt-get install -y -q cmake libboost-dev libtbb-dev

COPY . /app
WORKDIR /app/build
CMD cmake .. && make && ./cu_hlt -f ../input/minbias/velopix_raw -e ../input/minbias/ut_hits -d ../input/minbias/MC_info -g ../input/geometry/


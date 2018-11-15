FROM nvidia/cuda:10.0-devel-ubuntu18.04

RUN apt-get update -q -y && apt-get install -y -q cmake zlib1g-dev
COPY . /app
WORKDIR /app/build
CMD cmake .. && make && ./cu_hlt -f /input/minbias/banks/ -d /input/minbias/MC_info/ -g /input/detector_configuration/

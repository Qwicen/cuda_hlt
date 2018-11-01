FROM nvidia/cuda:10.0-devel-ubuntu18.04

RUN apt-get update -q
RUN apt-get install -y -q cmake libboost-dev libtbb-dev
COPY . /app
WORKDIR /app/build
CMD cmake .. && make &&	./cu_hlt 

FROM nvidia/cuda:10.0-devel

RUN apt-get update -q
RUN apt-get install -y -q cmake libboost-dev libtbb-dev
COPY . /app
WORKDIR /app/build
CMD cmake .. && make &&	./cu_hlt -f ../velopix_minbias_raw/ -g ../velopix_minbias_MC/ -t 3 -r 20 -n 1000 -c 0 

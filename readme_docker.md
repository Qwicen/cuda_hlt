The following lines will build the code base from any computer with NVidia-Docker, assuming you are in the directory with the code checkout and want to build in `build`:

```bash
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 --rm -v $(pwd):/cuda_hlt -it nvidia/cuda:9.2-devel-ubuntu18.04 bash

apt update && apt install -y cmake libtbb-dev
mkdir build
cmake ..
make
```

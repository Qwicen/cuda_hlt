The following lines will build the code base from any computer with NVidia-Docker, assuming you are in the directory with the code checkout and want to build in `build`:

First, build the docker image containing your code. If you change the code you should rebuild the image, so that it picks up the changes.

> Note: If you are working in a shared environment you might have a problem with a name collision, please consider adding $USER to the image name.

```bash
docker build -t $USER-cuda_hlt .
```

By default, this docker image would compile the code and run it with the input from the "/input" folder. In the command below we mount `input` inside this repository and mount the build folder, so that it caches built files.

```bash
docker run -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 --rm -v $(pwd)/build:/app/build -v $(pwd)/input:/input $USER-cuda_hlt
```

> Note: Files inside the build folder would belong to the root user.

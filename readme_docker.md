The following lines will build the code base from any computer with NVidia-Docker, assuming you are in the directory with the code checkout and want to build in `build`:
First you build docker image containing your code. Once you change your code you should rebuild the image, so it would pick up changes. 
Note: if you are working in a shared environment you might have a problem with a name collision, please consider adding $USER to a name for an image.
```bash
docker build -t $USER-cuda_hlt .
```
Default starting point for this docker image would compile code and run it with input from "/input" folder. In a command below we mount "input" inside this repository and mount build folder, so it would cache built files.
```bash
docker run -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 --rm -v $(pwd)/build:/app/build -v $(pwd)/input:/input $USER-cuda_hlt
```
Note: files inside build folder would belongs to a root user.

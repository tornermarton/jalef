# Docker configs for JALEF

*All commands must be run in the projects root folder!*

To update container please add required packages and tools to requirements.txt (python) and requirements.system 
(system - apt get ...) then rebuild and rerun container.

## Build container

If you consider to build your own docker image please use your own DockerHub username instead of *tornermarton*.
(also update all the other commands)

```bash
docker build -t tornermarton/tf-jupyter -f docker/Dockerfile .
```

## Run container

Starts the jupyter notebook server, tensorboard must be run separately. Sometimes Ctrl+P+Q does not work but closing the
terminal window is evenly good (will not kill the notebook process) just remember to write down the token for the first 
login.

```bash
./docker_run.sh
#OR
docker run --name 'tf-jupyter' -p 8888:8888 -p 6006:6006 -v $(pwd):/app tornermarton/tf-jupyter
```

## Run container on gpu-server (nvidia docker and CUDA required)

```bash
docker run --runtime=nvidia --name 'tmarton-jalef' -p 8888:8888 -p 6006:6006 -v $(pwd):/app tornermarton/tf-jupyter-gpu
```

## Start tensorboard 

```bash
docker exec -it tmarton-jalef bash -c "tensorboard --logdir /app/logs/tensorboard"
```
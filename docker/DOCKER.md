# Docker configs for JALEF

## Build container

```bash
# IN PROJECT ROOT!!!

docker build -t tornermarton/tf-jupyter -f docker/Dockerfile .
```

## Run container

```bash
./docker_run.sh
#OR
docker run --name 'tf-jupyter' -p 8888:8888 -p 6006:6006 -v $(pwd):/app tornermarton/tf-jupyter
```
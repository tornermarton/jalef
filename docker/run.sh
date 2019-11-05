#!/bin/bash

export UID=$(id -u)
export GID=$(id -g)

docker-compose -f docker-compose.yml up --build -d

echo "Notebook server:"
docker exec -it jalef bash -c "cat /var/log/supervisor/jupyter-notebook-stderr*" | grep token
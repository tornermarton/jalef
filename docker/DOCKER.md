# Docker configs for JALEF

## Start tensorflow 2.0 container with jupyter notebook

'''
docker run -it --name 'tf-jupyter' -p 8888:8888 -p 6006:6006 -v $(pwd)/notebooks:/notebooks -v $(pwd)/data:/data tensorflow/tensorflow:latest-py3-jupyter /bin/bash -c 'source /etc/bash.bashrc && jupyter notebook --notebook-dir=/notebooks --ip 0.0.0.0 --no-browser --allow-root'
'''
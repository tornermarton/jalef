FROM tensorflow/tensorflow:1.14.0-py3-jupyter
# tensorflow/tensorflow:latest-py3-gpu-jupyter

SHELL ["/bin/bash", "-c"]

WORKDIR /

COPY requirements.txt /
COPY requirements.system /

RUN apt-get update && pip install --upgrade pip && mkdir /app/
RUN xargs apt-get install < requirements.system
RUN pip install --trusted-host pypi.python.org -r ../requirements.txt

RUN apt-get install -y git

EXPOSE 8888
EXPOSE 6006

CMD ["bash", "-c", "source /etc/bash.bashrc & jupyter notebook --notebook-dir=/app/notebooks --ip 0.0.0.0 --no-browser --allow-root & tensorboard --logdir=/app/logs/tensorboard/"]
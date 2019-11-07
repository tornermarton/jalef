FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
MAINTAINER tornermarton

SHELL ["/bin/bash", "-c"]

WORKDIR /

COPY requirements.system /

# Install system packages
RUN apt -q update --fix-missing
RUN xargs apt -q install -y < requirements.system

# Create main app directory
RUN mkdir /app

# Configure ssh server
RUN mkdir /var/run/sshd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin no/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

# Change the initial root password,
# THIS MUST BE CHANGED IMMIDIATELY AFTER FIRST LOGIN!
RUN echo 'root:nehezjelszo' | chpasswd

# configure supervisor
COPY copy_files/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

COPY copy_files/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Install Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /usr/miniconda3
ENV PATH="/usr/miniconda3/bin:${PATH}"

# build checkpoint
COPY copy_files/environments /root/copy_files/environments
RUN for env in /root/copy_files/environments/*.yml; do conda env create -f $env; done
RUN conda init bash
# RUN conda env export -n tensorflow_2 > /root/copy_files/environments/backup_tensorflow_2.yml

# RUN pip install --trusted-host pypi.python.org -r ../requirements.txt

# correct cufflinks imports (plotly.plotly deprecated)
# RUN sed -i -e 's/plotly\.plotly/chart_studio\.plotly/g' /usr/local/lib/python3.6/dist-packages/cufflinks/tools.py \
#    && sed -i -e 's/plotly\.plotly/chart_studio\.plotly/g' /usr/local/lib/python3.6/dist-packages/cufflinks/plotlytools.py \
#    && sed -i -e 's/plotly\.plotly/chart_studio\.plotly/g' /usr/local/lib/python3.6/dist-packages/cufflinks/__init__.py

COPY copy_files/bashrc_append.txt /root/copy_files/bashrc_append.txt

EXPOSE 8888
EXPOSE 6006
EXPOSE 22

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
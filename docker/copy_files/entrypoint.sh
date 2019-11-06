#!/bin/bash

# Add local user
# Either use the LOCAL_USER_ID if passed in at runtime or
# fallback

USER_ID=${LOCAL_USER_ID:-9001}
USER_NAME=${USER_NAME}
GROUP_ID=${LOCAL_GROUP_ID:-9001}

echo "Starting with UID : $USER_ID"
groupadd -f -g $GROUP_ID $USER_NAME
useradd --shell /bin/bash -u $USER_ID -g $GROUP_ID -o -c "" -m $USER_NAME
adduser $USER_NAME sudo
echo $USER_NAME':nehezjelszo' | chpasswd
export HOME=/home/$USER_NAME

less /root/copy_files/bashrc_append.txt >> /root/.bashrc
less /root/copy_files/bashrc_append.txt >> /home/$USER_NAME/.bashrc

chown $USER_NAME:$USER_NAME /home/$USER_NAME/ -R

/usr/bin/supervisord
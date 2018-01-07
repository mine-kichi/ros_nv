#!/bin/bash

nvidia-docker run \
-e DISPLAY=172.17.0.1$DISPLAY \
-v /mnt/HDD/work:/home/mine/work \
--rm=true \
-it \
ros_indigo:test

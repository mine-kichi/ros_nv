#!/bin/bash

nvidia-docker run \
-e http_proxy=http://172.17.0.1:3128/ \
-e https_proxy=http://172.17.0.1:3128/ \
-e DISPLAY=172.17.0.1$DISPLAY \
-v /home/j0115858/work:/home/mine/work \
--rm=true \
-it \
ros_indigo:test

FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu14.04

LABEL Description="Ubuntu 14.04 dev env" Version="1.0"
ARG DEBIAN_FRONTEND=noninteractive
ARG DEBIAN_PRIORITY=critical
ENV TERM=xterm-256color

RUN	apt-get update && apt-get -y install firefox \
	language-pack-ja-base language-pack-ja \
	language-pack-gnome-ja \
	fonts-takao-gothic
ENV	LANG=ja_JP.UTF-8
RUN	apt-get -y update && apt-get -y install gedit \
	emacs
RUN	apt-get -y update && apt-get install -y sudo less


# Install Prerequisites
RUN apt-get -y update && apt-get install -y --no-install-recommends p7zip-full dpkg-dev 
RUN apt-get -y update && apt-get install -y --no-install-recommends lib32z1 tftp lib32ncurses5 lib32bz2-1.0 lib32stdc++6 isc-dhcp-server 
RUN apt-get -y update && apt-get install -y --no-install-recommends nfs-kernel-server minicom resolvconf gcc-multilib software-properties-common 
RUN apt-get -y update && apt-get install -y --no-install-recommends x11-utils libgtk2.0-0 libcanberra-gtk* cmake unzip expect usbutils dmidecode curl tree 
RUN apt-get -y update && apt-get install -y --no-install-recommends libreadline-dev build-essential libncurses5-dev git vim-gtk libtbb-dev libglew-dev libtiff5-dev 
RUN apt-get -y update && apt-get install -y --no-install-recommends libjpeg-dev libpng12-dev libjasper-dev libavcodec-dev libavformat-dev libavutil-dev libpostproc-dev 
RUN apt-get -y update && apt-get install -y --no-install-recommends libswscale-dev libeigen3-dev pkg-config software-properties-common
RUN	cp /usr/bin/sudo /usr/bin/pkexec	
RUN	bash -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN	apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116
RUN	apt-get update
RUN	apt-get install -y ros-indigo-desktop-full
RUN groupadd -g 1000 developer && \
    useradd  -g      developer -G sudo -m -s /bin/bash mine && \
    echo 'mine:mine' | chpasswd

RUN echo 'Defaults visiblepw'             >> /etc/sudoers
RUN echo 'mine ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN	cp /usr/bin/sudo /usr/bin/pkexec	

USER mine

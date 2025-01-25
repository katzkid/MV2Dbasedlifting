#!/bin/bash
# Update package list
sudo apt update

# Install the libGL library
sudo apt install -y libgl1-mesa-glx

# Install conda env
conda env create -f MV2Denv.yml

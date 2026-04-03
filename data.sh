#!/bin/sh

# Update system
sudo apt-get update

# Install squashfs
sudo apt-get install squashfs-tools

# Create squashfs dataset
# mksquashfs ./dataset/FloodNet/FloodNet-Supervised_v1.0 ~/flood_data.squashfs -comp zstd
mksquashfs ./dataset/FloodNet/FloodNet-Supervised_v1.0 ~/flood_data.squashfs

# Create mount point
mkdir -p ./dataset/data_mount

# Mount squashfs dataset
sudo mount -t squashfs ~/flood_data.squashfs ./dataset/data_mount

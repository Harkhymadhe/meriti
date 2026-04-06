#!/bin/sh

# # Update system
# sudo apt-get update

# # Install squashfs
# sudo apt-get install squashfs-tools

MOUNT_DIR=./dataset/data_mount
SQUASHFS_FILE=~/flood_data.squashfs

CREATE_MOUNT=$(findmnt -M $MOUNT_DIR && echo 0 || echo 1)
CREATE_SQUASHFS=$(test -e $SQUASHFS_FILE && echo 0 || echo 1)

echo "Create SQUASHFS file?: $CREATE_SQUASHFS"
echo "Create mount point?: $CREATE_MOUNT"


if [ $CREATE_SQUASHFS ]; then
    # Create squashfs dataset
    mksquashfs ./dataset/FloodNet/FloodNet-Supervised_v1.0 $SQUASHFS_FILE
fi

if [ $CREATE_MOUNT ]; then
    # Create mount point
    mkdir -p $MOUNT_DIR
    
    # Mount squashfs dataset
    sudo mount -t squashfs $SQUASHFS_FILE $MOUNT_DIR
fi

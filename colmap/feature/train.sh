#!/bin/bash

DATASET_PATH=/home/sjtu_dzn/NeRF/project

colmap automatic_reconstructor \
    --workspace_path $DATASET_PATH \
    --image_path $DATASET_PATH/images

#!/bin/bash

# 将变量 DATASET_PATH 设置为存储数据集（包括图像、数据库等）的路径。
DATASET_PATH=/home/sjtu_dzn/NeRF/project

# 从位于 $DATASET_PATH/images 的图像中提取特征，并将结果存储在 $DATASET_PATH/database.db 中。
colmap feature_extractor \
   --database_path $DATASET_PATH/database.db \
   --image_path $DATASET_PATH/images

# 在数据库中的所有图像对之间进行特征匹配。
colmap exhaustive_matcher \
   --database_path $DATASET_PATH/database.db

# 创建稀疏重建目录
mkdir $DATASET_PATH/sparse

# 执行稀疏三维重建（相机姿态和稀疏点云），并将结果保存到 $DATASET_PATH/sparse 目录中。
colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse

mkdir $DATASET_PATH/dense

# 使用稀疏重建过程中估计的相机姿态和内参对图像进行去畸变处理。去畸变后的图像将保存到 $DATASET_PATH/dense 目录中。
colmap image_undistorter \
    --image_path $DATASET_PATH/images \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/dense \
    --output_type COLMAP \
    --max_image_size 2000

# 使用PatchMatch算法进行稠密立体匹配，以创建稠密点云。
colmap patch_match_stereo \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

# 将通过立体匹配生成的稠密点云融合成一个统一的点云，并保存为 fused.ply。
colmap stereo_fusion \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $DATASET_PATH/dense/fused.ply

# 使用泊松表面重建方法从稠密点云生成3D网格。
colmap poisson_mesher \
    --input_path $DATASET_PATH/dense/fused.ply \
    --output_path $DATASET_PATH/dense/meshed-poisson.ply

# 对稠密点云数据进行 Delaunay 网格化，并将网格保存。
colmap delaunay_mesher \
    --input_path $DATASET_PATH/dense \
    --output_path $DATASET_PATH/dense/meshed-delaunay.ply
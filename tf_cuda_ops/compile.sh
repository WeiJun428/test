#!/usr/bin/env bash
# https://stackoverflow.com/a/10383546

set -e    # Exit immediately if a command exits with a non-zero status

if [ ! -d ./build ]; then
  mkdir build
fi
# NEED TO SWITCH TO THE CORRECT PYTHON ENVIRONMENT BEFORE EXECUTION.
TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

nvcc src/get_roi_bbox.cu -o build/get_roi_bbox.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -gencode=arch=compute_72,code=compute_72 -O3 
g++ -std=c++11 src/get_roi_bbox.cpp build/get_roi_bbox.cu.o -o build/get_roi_bbox.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
echo "Done: get_roi_bbox.so"

nvcc src/grid_sampling.cu -o build/grid_sampling.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -gencode=arch=compute_72,code=compute_72 -O3 
g++ -std=c++11 src/grid_sampling.cpp build/grid_sampling.cu.o -o build/libgrid_sampling.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
echo "Done: grid_sampling.so"

nvcc src/roi_logits_to_attrs.cu -o build/roi_logits_to_attrs.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -gencode=arch=compute_72,code=compute_72 -O3 
g++ -std=c++11 src/roi_logits_to_attrs.cpp build/roi_logits_to_attrs.cu.o -o build/roi_logits_to_attrs.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
echo "Done: roi_logits_to_attrs.so"

nvcc src/nms.cu -o build/nms.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -gencode=arch=compute_72,code=compute_72 -O3 
g++ -std=c++11 src/nms.cpp build/nms.cu.o -o build/nms.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
echo "Done: nms.so"

nvcc src/voxel_sampling_idx.cu -o build/voxel_sampling_idx.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -gencode=arch=compute_72,code=compute_72 -O3 
g++ -std=c++11 src/voxel_sampling_idx.cpp build/voxel_sampling_idx.cu.o -o build/voxel_sampling_idx.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
echo "Done: voxel_sampling_idx.so"

nvcc src/voxel_sampling_feature.cu -o build/voxel_sampling_feature.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -gencode=arch=compute_72,code=compute_72 -O3 
g++ -std=c++11 src/voxel_sampling_feature.cpp build/voxel_sampling_feature.cu.o -o build/voxel_sampling_feature.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
echo "Done: voxel_sampling_feature.so"

nvcc src/voxel_sampling_idx_binary.cu -o build/voxel_sampling_idx_binary.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -gencode=arch=compute_72,code=compute_72 -O3 
g++ -std=c++11 src/voxel_sampling_idx_binary.cpp build/voxel_sampling_idx_binary.cu.o -o build/voxel_sampling_idx_binary.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
echo "Done: voxel_sampling_idx_binary.so"

nvcc src/radix_sort1d.cu.cc -o build/radix_sort1d.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -gencode=arch=compute_72,code=compute_72 -O3 -I ../../../dependencies/thrust # -arch=sm_72 -O3 #
g++ src/radix_sort1d.cc build/radix_sort1d.cu.o -o build/radix_sort1d.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include -D GOOGLE_CUDA=1
echo "Done: radix_sort1d.so"

echo "Compile Tensorflow custom CUDA ops: Done"

#!/usr/bin/env bash

$nvcc tf_sampling_g.cu -c -o tf_sampling_g.cu.o  -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11  -I /home/gg1n1nder/.conda/envs/tensorflow/lib/python3.5/site-packages/tensorflow/include -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

g++ tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -std=c++11 -shared -fPIC -I /home/gg1n1nder/.conda/envs/tensorflow/lib/python3.5/site-packages/tensorflow/include -I/home/gg1n1nder/.conda/envs/tensorflow/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I = /opt/ohpc/pub/nvidia/cuda/cuda-9.0/include/ -L/home/gg1n1nder/.conda/envs/tensorflow/lib/python3.5/site-packages/tensorflow -ltensorflow_framework -lcudart -L /opt/ohpc/pub/nvidia/cuda/cuda-9.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0
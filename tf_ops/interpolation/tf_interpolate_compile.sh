# TF1.2
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /home/gg1n1nder/.conda/envs/tensorflow/lib/python3.5/site-packages/tensorflow/include -I /opt/ohpc/pub/nvidia/cuda/cuda-9.0/include -lcudart -L /opt/ohpc/pub/nvidia/cuda/cuda-9.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4
g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /home/gg1n1nder/.conda/envs/tensorflow/lib/python3.5/site-packages/tensorflow/include -I /opt/ohpc/pub/nvidia/cuda/cuda-9.0/include -I /home/gg1n1nder/.conda/envs/tensorflow/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -lcudart -L /opt/ohpc/pub/nvidia/cuda/cuda-9.0/lib64/ -L/home/gg1n1nder/.conda/envs/tensorflow/lib/python3.5/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

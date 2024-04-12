export IRTDM_CUDNN_LIBRARY=/usr/local/cuda/lib64
export IRTDM_CUDNN_INCLUDE_PATH=/usr/local/cuda/include
export IRTDM_LIBTORCH_DIR=/data/lib/libtorch

echo "Configuring and building irtdm ..."

mkdir build
cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" \
    -DCUDNN_LIBRARY=$IRTDM_CUDNN_LIBRARY \
    -DCUDNN_INCLUDE_PATH=$IRTDM_CUDNN_INCLUDE_PATH \

make -j2

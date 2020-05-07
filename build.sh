#!/bin/bash
rm -r ~/computer_vision/build && mkdir ~/computer_vision/build
source '/opt/intel/openvino/bin/setupvars.sh' && \
cd ~/computer_vision/build && \
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=armv7-a" .. && \
make -j4

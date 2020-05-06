MODE=$1
RUN=$2

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=armv7-a" .
make -j2
./bin/smart_camera
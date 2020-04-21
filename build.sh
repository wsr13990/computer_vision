MODE=$1
RUN=$2

if [ -n "$MODE" ];then
	MODE="Release"
fi

cmake -DCMAKE_BUILD_TYPE=$MODE -DCMAKE_CXX_FLAGS="-march=armv7-a" .
make -j2

if [ $RUN = "run" ];then
	./bin/smart_camera
fi

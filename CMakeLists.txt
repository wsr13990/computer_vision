cmake_minimum_required(VERSION 2.8.12)
set(PROJECT_NAME smart_camera)

PROJECT (${PROJECT_NAME})

# Enable C++11
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (DCMAKE_BUILD_TYPE Release)
set (DCMAKE_CXX_FLAGS "-march=armv7-a")
set (DTHREADING SEQ)

# find_package(OpenCV 4.1.1 EXACT REQUIRED QUIET )
find_package(InferenceEngine REQUIRED QUIET )
find_package(Boost REQUIRED filesystem QUIET )
find_package(gflags REQUIRED )

find_library(LIBX264 REQUIRED)
find_package(PkgConfig REQUIRED)

# message(PRINT ${OpenCV_INCLUDE_DIRS})
# message(PRINT ${OpenCV_LIBRARIES})

find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h)
find_library(AVCODEC_LIBRARY avcodec)

find_path(AVFORMAT_INCLUDE_DIR libavformat/avformat.h)
find_library(AVFORMAT_LIBRARY avformat)

find_path(AVUTIL_INCLUDE_DIR libavutil/avutil.h)
find_library(AVUTIL_LIBRARY avutil)

find_path(AVDEVICE_INCLUDE_DIR libavdevice/avdevice.h)
find_library(AVDEVICE_LIBRARY avdevice)

pkg_check_modules(GST REQUIRED gstreamer-1.0>=1.8)

# message(PRINT ${GST_INCLUDE_DIRS})
# message(PRINT ${GST_LIBRARIES})

if (${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
endif()

# Add project header & sources
file( GLOB LIB_SOURCES src/*.cpp)
file( GLOB LIB_HEADERS include/*.hpp)
# file( GLOB TBB_HEADER /usr/local/include/tbb/*.h)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH  ${CMAKE_CURRENT_BINARY_DIR}/bin)


# Create executable file from sources
add_executable(${PROJECT_NAME} ${LIB_SOURCES} ${LIB_HEADERS} )

# if(IE_SAMPLE_OPENCV_DEPENDENCIES)
# 	target_compile_definitions(${PROJECT_NAME} PRIVATE USE_OPENCV)
# endif()

target_include_directories( ${PROJECT_NAME} PRIVATE
	# ${TBB_INCLUDE}
	${InferenceEngine_INCLUDE_DIRS}
	# ${OpenCV_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS}
	# ${GST_INCLUDE_DIRS}
	${AVCODEC_INCLUDE_DIR}
	${AVFORMAT_INCLUDE_DIR}
	${AVUTIL_INCLUDE_DIR}
	${AVDEVICE_INCLUDE_DIR}
	sample_video
	sample_photo
	model
	embedding
	"/home/pi/opencv_install/include/opencv4"
)

target_link_directories( ${PROJECT_NAME} PRIVATE  ${EXTERNAL_LIB}
	"/home/pi/opencv_install/lib"
)

target_link_libraries(${PROJECT_NAME} PRIVATE
	# ${OpenCV_LIBRARIES}
	${InferenceEngine_LIBRARIES}
	${IE_RELEASE_LIBRARY}
	${Boost_LIBRARIES}
	# ${GST_LIBRARIES}
	${AVCODEC_LIBRARY}
	${AVFORMAT_LIBRARY}
	${AVUTIL_INCLUDE_DIR}
	${AVDEVICE_LIBRARY}
        gflags
	# IE::ie_cpu_extension
	opencv_aruco        opencv_dnn_objdetect  opencv_fuzzy      opencv_line_descriptor   opencv_quality    opencv_structured_light  opencv_videostab
	opencv_bgsegm       opencv_dnn            opencv_gapi       opencv_ml                opencv_reg        opencv_superres          opencv_xfeatures2d
	opencv_bioinspired  opencv_dpm            opencv_hfs        opencv_objdetect         opencv_rgbd       opencv_surface_matching  opencv_ximgproc
	opencv_face           opencv_highgui    opencv_optflow           opencv_saliency   opencv_text              opencv_xobjdetect
	opencv_features2d     opencv_imgcodecs  opencv_phase_unwrapping  opencv_shape      opencv_tracking          opencv_xphoto
	opencv_core         opencv_flann          opencv_img_hash   opencv_photo             opencv_stereo     opencv_videoio
	opencv_datasets     opencv_freetype       opencv_imgproc    opencv_plot              opencv_stitching  opencv_video
)
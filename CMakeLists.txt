cmake_minimum_required(VERSION 2.8)
set(PROJECT_NAME computer_vision)

PROJECT (${PROJECT_NAME})

find_package(OpenCV REQUIRED )
if(NOT(OpenCV_FOUND))
    message(WARNING "OPENCV is disabled or not found, " ${TARGET_NAME} " skipped")
    return()
endif()

set( OPENVINO_DIR "D:/Program Files (x86)/IntelSWTools/openvino")
set( OPENCV_DIR "D:/opencv")
set( BOOST_DIR "D:/C++Library/boost_1_72_0")
set( CMAKE_BINARY_DIR "D:/BELAJAR/C++/facial_recognition/")
set( MAIN_SRC src/main.cpp )
set( MSVC_DIR "D:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.23.28105" )

set( OPENVINO_2019_DIR "D:/Program Files (x86)/IntelSWTools/openvino_2019.3.379")
set( OPENVINO_2020_DIR "D:/Program Files (x86)/IntelSWTools/openvino_2020.1.033" )
set( IE_2019_DIR ${OPENVINO_2019_DIR}/deployment_tools/inference_engine)
set( OPENCV_BUILD "D:/opencv/build")
set( OPENCV4 "D:/OpenCV_Source/opencv-4.2.0")
set( OPENCV_CONTRIB "D:/OpenCV_Source/opencv_contrib")

MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
      LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()


SUBDIRLIST(SUBDIRS ${OPENCV4}/modules)
FOREACH(subdir ${SUBDIRS})
	message(STATUS ${OPENCV4}/modules/${subdir}/include)
	INCLUDE_DIRECTORIES(${OPENCV4}/modules/${subdir}/include)
ENDFOREACH()

SUBDIRLIST(SUBDIRS ${OPENCV_CONTRIB}/modules)
FOREACH(subdir ${SUBDIRS})
	message(STATUS ${OPENCV_CONTRIB}/modules/${subdir}/include)	
	INCLUDE_DIRECTORIES(${OPENCV_CONTRIB}/modules/${subdir}/include)
ENDFOREACH()

INCLUDE_DIRECTORIES(${IE_2019_DIR}/external/tbb/include
	${IE_2019_DIR}/src/extension
	${IE_2019_DIR}/demos/common/monitors
	${IE_2019_DIR}/demos/thirdparty/gflags/include
	${OPENCV_BUILD}
	${OPENCV4}/include)

INCLUDE_DIRECTORIES(${OPENVINO_2019_DIR}/deployment_tools/inference_engine/include
	"D:/C++Library/boost_1_72_0"
	sample_video)

link_directories( D:/C++Library/boost_1_72_0/stage/lib
	${OPENCV_DIR}/install/x64/vc16/lib
	D:/opencv/build/lib/Release
	D:/opencv/build/install/x64/vc16/bin )

# Add project header & sources
file( GLOB LIB_SOURCES src/*.cpp)
file( GLOB LIB_HEADERS include/*.hpp)

# Add Openvino library
file( GLOB OPENVINO_HEADER ${OPENCV_DIR}/inference_engine/include/*.hpp)

add_library( cv_lib ${LIB_SOURCES} ${LIB_HEADERS} ${OPENVINO_HEADER} )

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable( ${PROJECT_NAME} ${MAIN_SRC} ${LIB_SOURCES} ${LIB_HEADERS} )

set( OPENVINO_2019_RELEASE ${OPENVINO_2019_DIR}/deployment_tools/inference_engine/lib/intel64/Release/ )
set( OPENVINO_2020_IE ${OPENVINO_2020_DIR}/deployment_tools/inference_engine/ )
set( OPENVINO_2020_RELEASE ${OPENVINO_2020_IE}/lib/intel64/Release/ )

target_link_libraries( ${PROJECT_NAME}
	${OPENVINO_2020_IE}/demos/intel64/Release/cpu_extension.lib
	${OPENCV_BUILD}/lib/Release/opencv_highgui420.lib
	${OPENVINO_2019_RELEASE}/inference_engine.lib
	${OPENVINO_2020_IE}/demos/intel64/Release/monitors.lib
	${OPENVINO_2020_IE}/demos/intel64/Release/gflags_nothreads_static.lib
	${OPENVINO_2019_RELEASE}/tbb.lib
	"opencv_videoio420.lib"
	"opencv_imgcodecs420.lib"
	"opencv_imgproc420.lib"
	"opencv_core420.lib"
	"opencv_tracking420.lib"
	"opencv_objdetect420.lib"
	"opencv_highgui420.lib"
	"pdh.lib"
	"shlwapi.lib"
	"kernel32.lib"
	"user32.lib"
	"gdi32.lib"
	"winspool.lib"
	"shell32.lib"
	"ole32.lib"
	"oleaut32.lib"
	"uuid.lib"
	"comdlg32.lib"
	"advapi32.lib")
cmake_minimum_required(VERSION 3.6)
project(OpenCVStudy)

set(CMAKE_CXX_STANDARD 11)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

set(Tesseract_DIR /home/xsl/Applications/tesseract-master/build)
set(Leptonica_DIR /home/xsl/Applications/leptonica-master/build)
set(OpenCV_DIR /home/xsl/Applications/opencv-3.4.0/build)

find_package(Tesseract REQUIRED)
find_package(Leptonica REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${Tesseract_INCLUDE_DIRS} ${Leptonica_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})



set(SOURCE_FILE main.cpp ImageOperations.cpp)
add_executable(OpenCVStudy ${SOURCE_FILE})
target_link_libraries(OpenCVStudy ${Tesseract_LIBRARIES} ${Leptonica_LIBRARIES} ${OpenCV_LIBS})
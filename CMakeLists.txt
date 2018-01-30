cmake_minimum_required(VERSION 3.6)
project(OpenCVStudy)

set(CMAKE_CXX_STANDARD 14)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

#windows clion 使用vs编译器不能调试(开发中)
if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_windows)
    set(Tesseract_DIR C:/Program Files \(x86\)/tesseract)
    set(Leptonica_DIR C:/Program Files \(x86\)/leptonica)
    set(OpenCV_DIR C:/Tesseract-OCR/opencv-3.4.0/buildwithtesseract)
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(Tesseract_DIR /home/xsl/Applications/tesseract-master/build)
    set(Leptonica_DIR /home/xsl/Applications/leptonica-master/build)
    set(OpenCV_DIR /home/xsl/Applications/opencv-3.4.0/build)
endif ()
find_package(Tesseract REQUIRED)
find_package(Leptonica REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${Tesseract_INCLUDE_DIRS} ${Leptonica_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
#${Tesseract_LIBRARIES} ${Leptonica_LIBRARIES} ${OpenCV_LIBS}

set(SOURCE_FILE main.cpp ImageOperations.cpp)
add_executable(OpenCVStudy ${SOURCE_FILE})
target_link_libraries(OpenCVStudy ${Tesseract_LIBRARIES} ${Leptonica_LIBRARIES} ${OpenCV_LIBS})
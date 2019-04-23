cmake_minimum_required(VERSION 2.8)

project(opencv_qt)

find_package( OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} "balldetection.cpp")
target_link_libraries(${PROJECT_NAME}  ${OpenCV_LIBS})

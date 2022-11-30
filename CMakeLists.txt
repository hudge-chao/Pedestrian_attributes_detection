cmake_minimum_required(VERSION 2.8.3)

project(paddlepaddle_human_tracking)

# set(OpenCV_DIR /usr/lib/aarch64-linux-gnu/cmake/opencv4)

find_package(CUDA 10.2 REQUIRED)

find_package(OpenCV 3.4 REQUIRED)

message("opencv include dirs : ${OpenCV_INCLUDE_DIRS}")
message("cuda libraries dirs : ${CUDA_LIBRARIES}")
message("opencv libraries dirs : ${OpenCV_LIBRARIES}")

find_package(catkin REQUIRED COMPONENTS
	roscpp
	sensor_msgs
	std_msgs
	cv_bridge
	zed_interfaces
	message_filters
)

catkin_package(
	CATKIN_DEPENDS
	roscpp
	sensor_msgs
	std_msgs
	cv_bridge
	zed_interfaces
	message_filters
)

include_directories(
	${OpenCV_INCLUDE_DIRS}
	${CUDA_INCLUDE_DIRS}
	${catkin_INCLUDE_DIRS}
)

add_executable(inference src/inference.cc)

target_link_libraries(inference ${OpenCV_LIBRARIES} ${catkin_LIBRARIES} ${CUDA_LIBRARIES} nvinfer)

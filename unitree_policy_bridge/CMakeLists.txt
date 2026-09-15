cmake_minimum_required(VERSION 3.8)
project(unitree_policy_bridge)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

# Fill in with your local onnxruntime-linux-x64 release path. Same user-edited-placeholder
# convention as unitree_rl_lab's own UNITREE_ROS_DIR
# (source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py) - this repo makes no
# assumption about being checked out next to unitree-humanoidx, so there's nothing to derive
# this path from automatically.
if(NOT DEFINED ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_ROOT "/home/kieran/humanoidx/unitree-humanoidx/unitree_rl_lab/deploy/thirdparty/onnxruntime-linux-x64-1.22.0")
endif()
if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR
    "ONNX Runtime not found at '${ONNXRUNTIME_ROOT}'. Edit ONNXRUNTIME_ROOT near the top of "
    "this CMakeLists.txt to point at your onnxruntime-linux-x64 release directory, or pass "
    "-DONNXRUNTIME_ROOT=<path> to override without editing the file.")
endif()

# The vendored release only ships the versioned .so (libonnxruntime.so.1[.x]), no
# unversioned .so symlink - link whichever is actually present.
file(GLOB ONNXRUNTIME_SO "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so*")
list(GET ONNXRUNTIME_SO 0 ONNXRUNTIME_LIB)

add_executable(policy_bridge_node
  src/policy_bridge_node.cpp
)
target_include_directories(policy_bridge_node PRIVATE
  include
  ${ONNXRUNTIME_ROOT}/include
)
target_link_libraries(policy_bridge_node
  yaml-cpp
  ${ONNXRUNTIME_LIB}
)
ament_target_dependencies(policy_bridge_node
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
)

install(TARGETS policy_bridge_node
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()

cmake_minimum_required(VERSION 3.8)
project(unitree_isaac_policy)

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
find_package(spdlog REQUIRED)

# Three user-edited placeholders, same convention as unitree_policy_bridge's ONNXRUNTIME_ROOT:
# this repo makes no assumption about being checked out next to unitree-humanoidx, so there is
# nothing to derive these from. Override with -D<VAR>=<path> instead of editing if you prefer.
if(NOT DEFINED UNITREE_RL_LAB_ROOT)
  set(UNITREE_RL_LAB_ROOT "/home/kieran/humanoidx/unitree-humanoidx/unitree_rl_lab")
endif()
if(NOT DEFINED UNITREE_SDK2_ROOT)
  set(UNITREE_SDK2_ROOT "/home/kieran/humanoidx/unitree-humanoidx/unitree_sdk2")
endif()
if(NOT DEFINED ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_ROOT "${UNITREE_RL_LAB_ROOT}/deploy/thirdparty/onnxruntime-linux-x64-1.22.0")
endif()

set(ISAACLAB_INCLUDE "${UNITREE_RL_LAB_ROOT}/deploy/include")
if(NOT EXISTS "${ISAACLAB_INCLUDE}/isaaclab/envs/manager_based_rl_env.h")
  message(FATAL_ERROR
    "unitree_rl_lab's deploy headers not found under '${ISAACLAB_INCLUDE}'. Set "
    "UNITREE_RL_LAB_ROOT to your unitree_rl_lab checkout.")
endif()
# articulation.h includes unitree/dds_wrapper/common/unitree_joystick.hpp. Despite the path
# that header is standalone (math.h/stdint.h/iostream/chrono) and pulls in no DDS.
if(NOT EXISTS "${UNITREE_SDK2_ROOT}/include/unitree/dds_wrapper/common/unitree_joystick.hpp")
  message(FATAL_ERROR
    "unitree_joystick.hpp not found under '${UNITREE_SDK2_ROOT}/include'. Set UNITREE_SDK2_ROOT "
    "to your unitree_sdk2 checkout.")
endif()
if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR
    "ONNX Runtime not found at '${ONNXRUNTIME_ROOT}'. Set ONNXRUNTIME_ROOT to your "
    "onnxruntime-linux-x64 release directory.")
endif()

# The vendored release ships only the versioned .so, no unversioned symlink.
file(GLOB ONNXRUNTIME_SO "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so*")
list(GET ONNXRUNTIME_SO 0 ONNXRUNTIME_LIB)

add_executable(policy_node src/policy_node.cpp)
target_include_directories(policy_node PRIVATE
  ${ISAACLAB_INCLUDE}
  ${UNITREE_SDK2_ROOT}/include
  ${ONNXRUNTIME_ROOT}/include
)
target_link_libraries(policy_node yaml-cpp spdlog::spdlog ${ONNXRUNTIME_LIB})
ament_target_dependencies(policy_node rclcpp sensor_msgs geometry_msgs std_msgs)

add_executable(pd_node src/pd_node.cpp)
target_include_directories(pd_node PRIVATE include)
target_link_libraries(pd_node yaml-cpp)
ament_target_dependencies(pd_node rclcpp sensor_msgs std_msgs)

install(TARGETS policy_node pd_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()

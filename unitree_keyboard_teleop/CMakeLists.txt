cmake_minimum_required(VERSION 3.8)
project(unitree_keyboard_teleop)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(keyboard_handler REQUIRED)

add_executable(keyboard_controller src/keyboard_controller.cpp)
ament_target_dependencies(keyboard_controller rclcpp geometry_msgs keyboard_handler)

install(TARGETS keyboard_controller DESTINATION lib/${PROJECT_NAME})

ament_package()

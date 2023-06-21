cmake_minimum_required(VERSION 3.8)
project(slider_rl)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Python build
find_package(rclpy REQUIRED)


install(DIRECTORY
    trained_models
    DESTINATION share/${PROJECT_NAME}
)

# ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/slider_rl_controller.py
  DESTINATION lib/${PROJECT_NAME} 
)
ament_package()
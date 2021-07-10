cmake_minimum_required(VERSION 2.8.3)
project(pnc_task_msgs)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(catkin 
  REQUIRED COMPONENTS 
  message_generation 
  std_msgs 
  nav_msgs 
  rtp_msgs
  )

add_message_files(
  FILES
  ControlTask.msg
  PlanningTask.msg  
)

generate_messages(
  DEPENDENCIES
  std_msgs
  nav_msgs
  rtp_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime)
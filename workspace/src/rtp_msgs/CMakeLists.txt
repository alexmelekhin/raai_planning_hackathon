cmake_minimum_required(VERSION 2.8.3)
project(rtp_msgs)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(catkin COMPONENTS message_generation std_msgs geometry_msgs REQUIRED)

add_message_files(
  FILES
  RouteTask.msg
  RouteMetadata.msg
  PathPointMetadata.msg
  PathPointWithMetadata.msg
  PathStamped.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

catkin_package()

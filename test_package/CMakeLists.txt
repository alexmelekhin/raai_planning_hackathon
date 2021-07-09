cmake_minimum_required(VERSION 3.0.2)
project(test_package)

  add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  dynamic_reconfigure
)

generate_dynamic_reconfigure_options(
  config/Params.cfg
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp rospy
  DEPENDS  
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

ADD_DEFINITIONS( -std=c++11 )

file(GLOB_RECURSE test_package
  src/*
  )
add_executable(test_node ${test_package})


target_link_libraries(test_node
  ${catkin_LIBRARIES}
)

install(TARGETS test_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

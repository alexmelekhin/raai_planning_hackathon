#include "../include/test_class.h"
int main(int argc, char **argv)
{
  ros::init(argc, argv, "test_node");
  
  TestClass test;
  test.run();
  return 0;
}
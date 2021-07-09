#include "../include/test_class.h"

TestClass::TestClass()
{    
  test_pub = n.advertise<visualization_msgs::Marker>("/test_node", 1);
  
  test_sub = n.subscribe("/clicked_point", 1, &TestClass::test_callback, this);
}

void TestClass::run()
{
  dynamic_reconfigure::Server<test_package::ParamsConfig> r_server_;
  dynamic_reconfigure::Server<test_package::ParamsConfig>::CallbackType f = boost::bind(&TestClass::reconfigure_callback,
                                                                                                                    this, _1, _2);
  
  ros::Rate rate(10);
  
  while (n.ok())
  {
    ros::spinOnce();
    rate.sleep();
  }
}

//CALLBACKS

void TestClass::reconfigure_callback(test_package::ParamsConfig &config, uint32_t level)
{
  r = config.R;
}

void TestClass::test_callback(const geometry_msgs::PointStamped& point_stamped)
{
  ROS_ERROR("received point");

  visualization_msgs::Marker circle;

  circle.header.frame_id = "map";
  
  circle.type = 2;
  circle.pose.position.x = point_stamped.point.x;
  circle.pose.position.y = point_stamped.point.y;

  int radius;
  n.getParam("/test_node/R", radius);

  circle.scale.x = radius*2;
  circle.scale.y = radius*2;
  
  circle.color.r = 255;
  circle.color.a = 0.2;

  test_pub.publish(circle);
}

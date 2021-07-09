#include <iostream>

#include "ros/ros.h"
#include "dynamic_reconfigure/server.h"
#include "geometry_msgs/PointStamped.h"
#include "visualization_msgs/Marker.h";

#include "test_package/ParamsConfig.h"

using namespace std;
class TestClass
{
  private:
    
    ros::NodeHandle n; 
    ros::Publisher test_pub;
    ros::Subscriber test_sub;
    
    double r;

  public:
  
    TestClass();
    void run();
    void reconfigure_callback(test_package::ParamsConfig &config, uint32_t level);
    void test_callback(const geometry_msgs::PointStamped& point_stamped);
};
std_msgs/Header header

# Current robot position %
nav_msgs/Odometry current_pose

# Emergency stop %
bool estop

# Regular stop %
bool stop

# Trajectory %
rtp_msgs/PathStamped trajectory

# Map %
nav_msgs/OccupancyGrid occ_grid
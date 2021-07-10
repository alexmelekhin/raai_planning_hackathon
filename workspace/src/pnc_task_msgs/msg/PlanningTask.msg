std_msgs/Header header

# Init robot position %
rtp_msgs/PathPointWithMetadata init_point

# Goal robot position %
rtp_msgs/PathPointWithMetadata goal_point

# Map %
nav_msgs/OccupancyGrid occ_grid

# Options %
uint8 path_type
uint16 route_index_to       # Индекс точки маршрута, к которой построена траектория
uint16 route_index_from     # Индекс точки маршрута, от которой построена траектория
#!/usr/bin/env python
import rospy
from pnc_task_msgs.msg import PlanningTask
from rtp_msgs.msg import PathStamped, \
                         PathPointWithMetadata, \
                         PathPointMetadata, \
                         RouteTask
from geometry_msgs.msg import Pose

import numpy as np
import heapq


mock_grid_data = grid = [
    0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

mock_start_point = (1, 2)
mock_goal_point = (19, 1)


def convert_grid(map_data, map_h, map_w):
    result = np.zeros((map_h, map_w))
    for row in range(map_h):
        for col in range(map_w):
            if map_data[row*map_h + col] > 50:
                result[row][col] = 1
            else:
                result[row][col] = 0
    return result

#####
# A-star algorithm:
def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def astar(array, start, goal):
    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
 
    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]

            return data

        close_set.add(current)

        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:                
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue

                else:
                    # array bound y walls
                    continue

            else:
                # array bound x walls
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
 
            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return False

def pose2xy(pose, map_res):
	x = pose.position.x // map_res
	y = pose.position.y // map_res
	return x, y


def callback(task):
    """Route planning callback

    Parameters:
    task (PlanningTask): message with planning task

    """
    print("\nRecieved task")

    init_point = task.init_point  # rtp_msgs/PathPointWithMetadata
    goal_point = task.goal_point  # rtp_msgs/PathPointWithMetadata
    occ_grid = task.occ_grid  # nav_msgs/OccupancyGrid
    path_type = task.path_type
    route_index_to = task.route_index_to
    route_index_from = task.route_index_from

    start_pose = init_point.pose  # geometry_msgs/Pose
    goal_pose = goal_point.pose  # geometry_msgs/Pose

    map_res = occ_grid.info.resolution
    map_h = occ_grid.info.height
    map_w = occ_grid.info.width
    map_origin_pose = occ_grid.info.origin

    map_data = occ_grid.data
    map_data = convert_grid(map_data, map_h, map_w)

    start_point = task.route_index_from // map_w, task.route_index_from % map_w
    print(f"start point: {start_point}")

    goal_point = task.route_index_to // map_w, task.route_index_to % map_w
    print(f"goal point: {goal_point}")

    global pub
    path_stamped = PathStamped()
    path_stamped.header = task.header
    path_stamped.path_with_metadata = [init_point]
    path_stamped.path_type = task.path_type
    path_stamped.route_index_to = task.route_index_to
    path_stamped.route_index_from = task.route_index_from

    path_trajectory = astar(map_data, start_point, goal_point)

    path_trajectory = path_trajectory[::-1]

    if path_trajectory == False:
    	print("ERORE")

    for point in path_trajectory:
        new_pose = Pose()
        new_pose.position.x = point[0] * map_res
        new_pose.position.y = point[1] * map_res
        new_pose.position.z = 0.0

        new_pose.orientation.x = 0.0
        new_pose.orientation.y = 0.0
        new_pose.orientation.z = 0.0
        new_pose.orientation.w = 1.0

        path_point = PathPointWithMetadata()
        path_point.metadata = dfltPathPointMetadata()
        path_point.pose = new_pose

        path_stamped.path_with_metadata.append(path_point)

    path_stamped.path_with_metadata.append(task.goal_point)


    print(f"\n\nPATH TRAJECTORY:\n{path_trajectory}")

    print(path_stamped)

    pub.publish(path_stamped)

    print("Path published\n")


def callback2(route_task):
    pass

def dfltPathPointMetadata():
    """Returns default metadata for point"""
    metadata = PathPointMetadata()
    metadata.linear_velocity = 2.0
    metadata.max_deviation = 0.0
    metadata.key_point = False
    metadata.return_point = False
    metadata.return_point = False
    metadata.delay = 0.0
    return metadata


if __name__ == '__main__':
    rospy.init_node("route_planner")

    print("Started")
    pub = rospy.Publisher("/planner_sb_node/trajectory", PathStamped, queue_size=10)
    # pub = rospy.Publisher("/pl?anning_node/trajectory", PathStamped, queue_size=10)
    rospy.Subscriber("/planning_node/task", PlanningTask, callback)

    # rospy.Subscriber("/route_creator_node/route", RouteTask, callback2)

    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.05)
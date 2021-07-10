import sys
import numpy as np
import math
import random
from pyflann import *
import scipy.spatial
import time
import dubins

radius = 1.8 / map_res
flann = FLANN()
entries = []
locations = []
goal_point = []
nodes = list()

# Node representation
class Node(object):

	def __init__(self, parent, configuration, path):
		self.parent = parent
		self.x = configuration[0]
		self.y = configuration[1]
		self.theta = configuration[2]
		self.configuration = configuration
		self.location = (self.x, self.y)
		self.path = path

	def calcDepth(self):
		if self != root:
			return self.path.path_length() + self.parent.calcDepth()
		else:
			return 0


# tests the depth of node given a parent (both nodes)
def testDepth(node, givenParent):
	path = dubins.shortest_path(
	    givenParent.configuration, node.configuration, 0.2)
	return path.path_length() + givenParent.calcDepth()


# returns valid non-goal sample
def validSample():
	samplex = random.uniform(0, numColumns)
	sampley = random.uniform(0, numRows)
	sampletheta = random.uniform(0, 1000)

	_sample = np.array((samplex, sampley, sampletheta))
	worldItem = entries[int(sampley)][int(samplex)]

	if worldItem == "#": return validSample()
	else: return _sample


# given node, return all NODES within a given radius
def near(flann, node):
	flann.build_index(locations)
	indices = flann.nn_radius(np.array(node.location), radius)[0]
	ret = list()
	for index in indices:
		if nodes[index].location != node.location:
			ret.append(nodes[index])
		else:
			print('!')
	return ret


# returns a sample w/ goal bias
def sample():
	rand = random.randint(1, 200)
	if(rand == 20):
		return goal_point
	else:
		return validSample()


# get series of points made by dubins path of two full locations
# convert tot numpy array for slicing
# return sliced
def trajectory(q0, q1):
	path = dubins.shortest_path(q0, q1, 0.2)
	configurations, _ = path.sample_many(0.01)
	c = np.array(configurations)
	return c[:, :2]


# checks points along trajectory for collisions
# given list of points as floats
def validPath(trajectory):
	# print len(trajectory)
	for point in trajectory:
		discrX = int(point[0])
		discrY = int(point[1])
		try:
			worldItem = entries[discrY][discrX]
			if point[0] > numColumns or point[0] < 0.0 or point[1] > numRows or point[1] < 0.0: return False
			if worldItem == "#": return False
		# sometimes the path is out of bounds
		except IndexError:
			return False
	return True


# take target x, y, return closest node using flann
def nearest(x, y):
	flann.build_index(locations)
	dataset = locations
	testset = np.array((x, y))
	result, dists = flann.nn(
	    locations, testset, 1, algorithm="kmeans", branching=32, iterations=7, checks=16)
	return nodes[result[0]]


# Rapidly Exploring Random Tree
# returns solution node
def rrt():
	global locations
	while True:
		_sample = sample()
		_closestNode = nearest(_sample[0], _sample[1])
		if validPath(trajectory(_closestNode.configuration, _sample)):
			path = dubins.shortest_path(_closestNode.configuration, _sample, 0.2)
			new = Node(_closestNode, _sample, path)
			nodes.append(new)
			locations = np.concatenate((locations, [new.location]), axis=0)
			if new.x == goal_point[0] and new.y == goal_point[1]: return new


# actively rewire a neighborhood (nodes) considering center node
def rewire(neighborhood, center):
	for neighbor in neighborhood:
		if validPath(trajectory(center.configuration, neighbor.configuration)):
			if testDepth(neighbor, center) < neighbor.calcDepth():
				newpath = dubins.shortest_path(
				    center.configuration, neighbor.configuration, 0.2)
				neighborUpdate = Node(center, neighbor.configuration, newpath)
				nodes.remove(neighbor)
				nodes.append(neighborUpdate)
				assert nodes[nodes.index(neighborUpdate)].parent == center
				assert nodes[nodes.index(neighborUpdate)].path == newpath

	# for neighbor in neighborhood:
	# 	if validPath(trajectory(center.configuration, neighbor.configuration)):
	# 		if testDepth(neighbor, center) < neighbor.calcDepth():
	# 			neighbor.parent = center
	# 			neighbor.path = dubins.shortest_path(center.configuration, neighbor.configuration, 0.2)
	# 			nodes.append(neighbor)


# Rapidly Exploring Random Tree w/ rewiring
def rrt_star():
	global locations
	while True:
		_sample = sample()
		new = Node(None, _sample, None)
		neighborhood = near(flann, new)
		neighborhood.sort(key=lambda x: testDepth(new, x), reverse=False)
		if len(neighborhood) == 0: continue
		parent = neighborhood[0]
		if validPath(trajectory(parent.configuration, new.configuration)):
			new.parent = parent
			newP = dubins.shortest_path(parent.configuration, new.configuration, 0.2)
			new.path = newP
			assert new.parent == parent
			assert new.path == newP
			nodes.append(new)
			locations = np.concatenate((locations, [new.location]), axis=0)
			rewire(neighborhood, new)
			if new.x == goal_point[0] and new.y == goal[1]: return new



def plan_trajectory(start, goal, map_data, map_w, map_h, map_res):
	
	algorithm = rrt_star
	global radius #= 1.8 / map_res

	# libs
	global flann #= FLANN()
	# np.set_printoptions(threshold=np.nan)

	# stdin processing
	global entries
	global numColumns
	numColumns = map_w
	global numRows
	numRows = map_h
	map_data = np.array(map_data)
	map_data = map_data.reshape((map_h, map_w))
	formatted_map = []
	for row in map_data:
		new_row = []
		for cell in row:
			if cell > 50:
				new_row.append('#')
			else:
				new_row.append(' ')
		formatted_map("".join(new_row))

	entries = formatted_map

	# setup
	start_point = float(start[0]), float(start[1])
	root = Node(None, np.array((start_point[0], start_point[1], 1.57)), None)
	global goal_point 
	goal_point = np.array((float(goal[0]), float(goal[1]), 0))
	global nodes 
	nodes = list()
	nodes.append(root)
	global locations 
	locations = np.array([root.location])

	# Path planning
	startTime = time.time()
	solution = algorithm()
	duration = time.time() - startTime

	# reporting for analysis
	# print solution.calcDepth() , ',' , duration

	# solution reporting for visualization
	solutionTrajectory = list()
	cur = solution
	while cur != root:
		configurations, _ = cur.path.sample_many(0.1)
		for c in reversed(configurations):
			solutionTrajectory.append((c[0], c[1]))
		cur = cur.parent

	print("\n\nSOLUTION\n")

	print(len(solutionTrajectory))

	for point in solutionTrajectory:
		print(point[0], point[1])

	return solutionTrajectory
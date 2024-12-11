#!/usr/bin/env python3
"""
ROS Node for Autonomous Navigation of a Multi TurtleBots Using ArUco Markers

This script navigates a TurtleBot along a predefined path by reading waypoints from a YAML file.
It uses ArUco markers for localization and computes the required transformations to map simulation
coordinates to real-world coordinates.


Requirements:
- ROS (Robot Operating System)
- OpenCV with ArUco module
- PyYAML
"""

import rospy
import math
from geometry_msgs.msg import Twist, PoseStamped
from tf.transformations import euler_from_quaternion
import yaml
import numpy as np
import cv2
import threading
from cbs import cbs
from sensor_msgs.msg import LaserScan


def convert_sim_to_real_pose(x, y, matrix):
    """
    Converts simulation coordinates to real-world coordinates using a perspective transformation matrix.

    Parameters:
    - x (float): X-coordinate in simulation.
    - y (float): Y-coordinate in simulation.
    - matrix (np.ndarray): 3x3 perspective transformation matrix.

    Returns:
    - Tuple[float, float]: Transformed X and Y coordinates in real-world.
    """
    # Create a homogeneous coordinate for the point
    point = np.array([x, y, 1])

    # Apply the perspective transformation
    transformed_point = np.dot(matrix, point)

    # Normalize to get the actual coordinates
    transformed_point = transformed_point / transformed_point[2]

    return transformed_point[0], transformed_point[1]


def check_goal_reached(current_pose, goal_x, goal_y, tolerance):
    """
    Checks if the robot has reached the goal position within a specified tolerance.

    Parameters:
    - current_pose (PoseStamped): Current pose of the robot.
    - goal_x (float): Goal X-coordinate.
    - goal_y (float): Goal Y-coordinate.
    - tolerance (float): Acceptable distance from the goal to consider it reached.

    Returns:
    - bool: True if goal is reached, False otherwise.
    """
    # Get current position
    current_x = current_pose.pose.position.x
    current_y = current_pose.pose.position.y

    # Check if within tolerance
    if (abs(current_x - goal_x) <= tolerance and abs(current_y - goal_y) <= tolerance):
        return True
    else:
        return False


def navigation(turtlebot_name, aruco_id, goal_list, barrier=None):
    """
    Navigates the TurtleBot through a list of waypoints.

    Parameters:
    - turtlebot_name (str): Name of the TurtleBot.
    - aruco_id (str): ArUco marker ID used for localization.
    - goal_list (List[Tuple[float, float]]): List of (X, Y) coordinates as waypoints.
    """
    current_position_idx = 0  # Index of the current waypoint

    # Publisher to send velocity commands to the robot
    # 创建一个ROS发布者cmd_pub，发布速度指令到主题/{turtlebot_name}/cmd_vel。
    cmd_pub = rospy.Publisher(f'/{turtlebot_name}/cmd_vel', Twist, queue_size=1)

    # Wait for the initial pose message from the ArUco marker
    # 等待从指定的ArUco标记主题/{aruco_id}/aruco_single/pose接收到初始位姿消息。
    init_pose = rospy.wait_for_message(f'/{aruco_id}/aruco_single/pose', PoseStamped, timeout=5)

    # Initialize Twist message for velocity commands
    twist = Twist()

    # Loop until all waypoints are reached or ROS is shut down
    while current_position_idx < len(goal_list) and not rospy.is_shutdown():
        # Get current goal coordinates
        goal_x, goal_y = goal_list[current_position_idx]

        # Check if the goal has been reached
        # 检查是否到达下一个节点，如果到达下一个节点继续检查是否到达最后一个节点
        if check_goal_reached(init_pose, goal_x, goal_y, tolerance=0.1):
            rospy.loginfo(f"{turtlebot_name}: Waypoint {current_position_idx + 1} reached: Moving to next waypoint.")
            current_position_idx += 1  # Move to the next waypoint

            # If all waypoints are reached, exit the loop
            if current_position_idx >= len(goal_list):
                rospy.loginfo(f"{turtlebot_name}: All waypoints have been reached.")
                break

        # Update the current pose
        init_pose = rospy.wait_for_message(f'/{aruco_id}/aruco_single/pose', PoseStamped, timeout=5)

        # Extract the current orientation in radians from quaternion
        orientation_q = init_pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        current_orientation = yaw  # Current heading of the robot

        # Calculate the difference between the goal and current position
        dx = goal_x - init_pose.pose.position.x
        dy = goal_y - init_pose.pose.position.y
        distance = math.hypot(dx, dy)  # Euclidean distance to the goal
        goal_direction = math.atan2(dy, dx)  # Angle to the goal

        # Normalize angles to range [0, 2π)
        current_orientation = (current_orientation + 2 * math.pi) % (2 * math.pi)
        goal_direction = (goal_direction + 2 * math.pi) % (2 * math.pi)

        # Compute the smallest angle difference
        theta = goal_direction - current_orientation

        # Adjust theta to be within [-π, π]
        if theta > math.pi:
            theta -= 2 * math.pi
        elif theta < -math.pi:
            theta += 2 * math.pi

        # Log debug information
        rospy.logdebug(f"Current Position: ({init_pose.pose.position.x:.2f}, {init_pose.pose.position.y:.2f})")
        rospy.logdebug(f"Goal Position: ({goal_x:.2f}, {goal_y:.2f})")
        rospy.logdebug(f"Current Orientation: {current_orientation:.2f} rad")
        rospy.logdebug(f"Goal Direction: {goal_direction:.2f} rad")
        rospy.logdebug(f"Theta (Angle to Goal): {theta:.2f} rad")
        rospy.logdebug(f"Distance to Goal: {distance:.2f} meters")

        # Control parameters (adjust these as needed)
        k_linear = 0.5    # Linear speed gain
        k_angular = 2.0   # Angular speed gain

        # Compute control commands
        linear_velocity = k_linear * distance * math.cos(theta)  # Move forward towards the goal
        angular_velocity = -k_angular * theta  # Rotate towards the goal direction

        # Limit maximum speeds if necessary
        max_linear_speed = 0.2  # meters per second
        max_angular_speed = 1.0  # radians per second

        linear_velocity = max(-max_linear_speed, min(max_linear_speed, linear_velocity))
        angular_velocity = max(-max_angular_speed, min(max_angular_speed, angular_velocity))

        # Set Twist message
        twist.linear.x = linear_velocity
        twist.angular.z = angular_velocity

        # Publish the velocity commands
        cmd_pub.publish(twist)

        # Sleep to maintain the loop rate
        rospy.sleep(0.1)  # Adjust the sleep duration as needed
    
     # 在第一段导航结束后，调用barrier等待其他代理
    if barrier is not None:
        rospy.loginfo(f"{turtlebot_name}: Waiting at barrier after stage 1.")
        barrier.wait()
        rospy.loginfo(f"{turtlebot_name}: Passed barrier, starting stage 2.")


def get_transformation_matrix(aruco_markers):
    """
    Detects corner ArUco markers and calculates the perspective transformation matrix.

    Parameters:
    - aruco_markers (List[str]): List of ArUco marker IDs used for the transformation.

    Returns:
    - np.ndarray: 3x3 perspective transformation matrix.
    """
    # Dictionary to store the poses of the ArUco markers
    marker_poses = {}

    # Wait for ArUco marker poses to define transformation between simulation and real-world coordinates
    # marker_poses词典记录aruco_markers列表中ArUco marker的观测位置
    for marker_id in aruco_markers:
        try:
            # Wait for the pose of each ArUco marker
            pose = rospy.wait_for_message(f'/{marker_id}/aruco_single/pose', PoseStamped, timeout=5)
            marker_poses[marker_id] = (pose.pose.position.x, pose.pose.position.y)
            rospy.loginfo(f"Received pose for marker {marker_id}: x={pose.pose.position.x}, y={pose.pose.position.y}")
        except rospy.ROSException:
            rospy.logerr(f"Timeout while waiting for pose of marker {marker_id}")
            raise

    # Define real-world and simulation points for the perspective transformation
    real_points = np.float32([
        marker_poses['id503'],  # Bottom-left corner in real world
        marker_poses['id502'],  # Bottom-right corner in real world
        marker_poses['id500'],  # Top-left corner in real world
        marker_poses['id501']   # Top-right corner in real world
    ])

    sim_points = np.float32([
        [0, 0],     # Bottom-left corner in simulation
        [10, 0],    # Bottom-right corner in simulation
        [0, 10],    # Top-left corner in simulation
        [10, 10]    # Top-right corner in simulation
    ])

    # Calculate the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(sim_points, real_points)

    rospy.loginfo("Perspective transformation matrix calculated successfully.")

    return matrix


def create_env(yaml_file):
    """
    Creates and loads assets only related to the environment such as boundaries and obstacles.
    Robots are not created in this function (check `create_turtlebot_actor`).
    """
    with open(yaml_file, 'r') as f:
        try:
            env_params = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as e:
            print(e) 
            
    rospy.loginfo("Environmental dict created successfully.")

    return env_params


def read_and_transform_waypoints(inputfile_path, matrix, env_params, agent_yaml_params_dict, outputfile_path):
    """
    Reads waypoints from a YAML file and transforms them from simulation to real-world coordinates.

    Parameters:
    - file_path (str): Path to the YAML file containing the schedule.
    - matrix (np.ndarray): Perspective transformation matrix.

    Returns:
    - List[Tuple[float, float]]: List of transformed waypoints.
    """
    def run_cbs(env_params, agent_yaml_params_dict, outputfile_path):
        """
        Runs the CBS algorithm to generate the schedule.
        """
        cbs.run(
            dimensions = env_params["map"]["dimensions"],
            obstacles = env_params["map"]["obstacles"],
            agents = agent_yaml_params_dict["agents"],
            out_file = outputfile_path
        )
    # Run the CBS algorithm
    try:
        run_cbs(env_params, agent_yaml_params_dict, outputfile_path)
    except Exception as e:
        rospy.logerr(f"Failed to run CBS algorithm: {e}")
        raise

    # Read the schedule from the YAML file
    def read_yaml_file(inputfile_path):
        """
        Reads the schedule from a YAML file.

        Parameters:
        - inputfile_path (str): Path to the YAML file.

        Returns:
        - dict: Dictionary containing the schedule data.
        """
        with open(inputfile_path, 'r') as file:
            data = yaml.safe_load(file)
        return data['schedule']  # Returns a dictionary of steps
    # try:
    #     # Load schedule data from YAML file
    #     schedule_data = read_yaml_file(inputfile_path)
    # except Exception as e:
    #     rospy.logerr(f"Failed to read schedule YAML file: {e}")
    #     raise

    # 读cbs输出的新schedule文件
    try:
        # Load schedule data from YAML file
        schedule_data = read_yaml_file(outputfile_path)
    except Exception as e:
        rospy.logerr(f"Failed to read schedule YAML file: {e}")
        raise

    coordinates = {}  # Dictionary to store transformed waypoints for each agent

    # Process waypoints for each agent
    for agent_id, steps in schedule_data.items():
        rospy.loginfo(f"Processing agent {agent_id}")
        agent_coordinates = []  # List to store waypoints for this agent

        for step in steps:
            # Simulation coordinates
            sim_x = step['x']
            sim_y = step['y']

            # Transform simulation coordinates to real-world coordinates
            real_x, real_y = convert_sim_to_real_pose(sim_x, sim_y, matrix)

            rospy.loginfo(f"Transformed simulation coordinates ({sim_x}, {sim_y}) to real-world coordinates ({real_x:.2f}, {real_y:.2f})")

            # Append the transformed coordinates to the list
            agent_coordinates.append((real_x, real_y))

        # break  # Remove this if you want to process multiple agents

        # Add the agent's coordinates to the dictionary
        # 键依旧为list
        coordinates[agent_id] = agent_coordinates

    return coordinates


def create_agent_yaml_params(agents, aruco_ids, aruco_goals_ids):
    rospy.loginfo("Executing create_agent_yaml_params\n")
    # Build agent_yaml_params["agents"]
    agent_yaml_params = {"agents": []}  # Initialize the agents list within the dictionary
    try:
        for agent_id in agents:
            rospy.loginfo(f"Creating agent {agent_id} yaml params. \n")
            # 获取每个代理的起始位置
            try:
                start_aruco_id = aruco_ids[agent_id]
                start_pose = rospy.wait_for_message(f'/{start_aruco_id}/aruco_single/pose', PoseStamped, timeout=5)
                start_x = start_pose.pose.position.x
                start_y = start_pose.pose.position.y
            except Exception as e:
                rospy.logerr(f"Failed to get start position for agent {agent_id}: {e}")
                raise
            # 获取每个代理的目标位置
            try:
                end_aruco_id = aruco_goals_ids[agent_id]
                end_pose = rospy.wait_for_message(f'/{end_aruco_id}/aruco_single/pose', PoseStamped, timeout=5)
                end_x = end_pose.pose.position.x
                end_y = end_pose.pose.position.y
            except Exception as e:
                rospy.logerr(f"Failed to get goal position for agent {agent_id}: {e}")
                raise

            # Build the agent dictionary
            agent_dict = {
                'goal': [end_x, end_y],
                'name': agent_id,
                'start': [start_x, start_y]
            }
            agent_yaml_params["agents"].append(agent_dict)
    except Exception as e:
        rospy.logerr(f"An error occurred while creating agent yaml params: {e}")
        raise

    # for agent_id in agents:
    #     rospy.loginfo(f"Creating agent {agent_id} yaml params. \n")
    #     # For each agent, get start positions via ArUco markers
    #     start_aruco_id = aruco_ids[agent_id]
    #     start_pose = rospy.wait_for_message(f'/{start_aruco_id}/aruco_single/pose', PoseStamped)
    #     start_x = start_pose.pose.position.x
    #     start_y = start_pose.pose.position.y
    #     rospy.loginfo(f"Got agent {agent_id} start position. \n")

    #     # For each agent, get goal positions via ArUco markers
    #     end_aruco_id = aruco_goals_ids[agent_id]
    #     end_pose = rospy.wait_for_message(f'/{end_aruco_id}/aruco_single/pose', PoseStamped)
    #     end_x = end_pose.pose.position.x
    #     end_y = end_pose.pose.position.y
    #     rospy.loginfo(f"Got agent {agent_id} end position. \n")

    #     # Build the agent dictionary
    #     agent_dict = {
    #         'goal': [end_x, end_y],
    #         'name': agent_id,
    #         'start': [start_x, start_y]
    #     }
    #     agent_yaml_params["agents"].append(agent_dict)

    return agent_yaml_params


def run(agents, turtlebot_names, aruco_ids, coordinates, stage):
    """
    Set up threads to publish velocity commands for each robot to reach goal positions.

    Parameters:
    - agents (list): List of agent IDs.
    - turtlebot_names (dict): Dictionary mapping agent IDs to turtlebot names.
    - aruco_ids (dict): Dictionary mapping agent IDs to ArUco IDs for localization.
    - coordinates (dict): Dictionary of waypoints for each agent.
    """
    threads = []

    if stage == 1:
        # 创建Barrier对象，参与的线程数为代理数量
        barrier = threading.Barrier(len(agents))
    else:
        barrier = None  # 第二段导航不需要barrier
    
    for agent_id in agents:
        t = threading.Thread(target=navigation, args=(turtlebot_names[agent_id], aruco_ids[agent_id], coordinates[agent_id], barrier))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()


def main():
    """
    Main function to initialize the ROS node and start the navigation process.
    """
    rospy.init_node('multi_robot_goal_pose_solutione')

    # List of ArUco marker IDs used for the transformation
    aruco_markers = ['id500', 'id501', 'id502', 'id503']

    try:
        # Get the transformation matrix using the corner detection function
        matrix = get_transformation_matrix(aruco_markers)
    except Exception as e:
        rospy.logerr(f"Failed to get transformation matrix: {e}")
        return

    # try:
    #     # Read and transform waypoints from the YAML file
    #     # 此时coordinates是一个列表，其中元素是(x,y)坐标
    #     coordinates = read_and_transform_waypoints("./cbs_output.yaml", matrix)
    # except Exception as e:
    #     rospy.logerr(f"Failed to read and transform waypoints: {e}")
    #     return

    # Start navigation with the first agent's waypoints
    # turtlebot_name = "turtle1"  # Name of your TurtleBot
    # aruco_id = "id402"          # ArUco marker ID for localization

    # List of agent IDs
    agents = [1, 2]
    # Dictionary of turtlebot names
    turtlebot_names = {
        1: 'turtle1',
        2: 'turtle2'
    }
    # Dictionary of ArUco IDs for localization
    aruco_ids = {
        1: 'id402',
        2: 'id404'  # 修改这个id
    }
    # Dictionary of ArUco IDs for waypoints
    aruco_waypoints_ids= {
        1: 'id111',
        2: 'id222'  # 修改这个id
    }
    # Dictionary of ArUco IDs for goalpoints
    aruco_goalpoints_ids= {
        1: 'id333',
        2: 'id444'  # 修改这个id
    }

    # Load environment parameters (只需加载一次)
    try:
        env_params = create_env("environment.yaml") # 请确保提供正确的环境配置文件路径
    except Exception as e:
        rospy.logerr(f"Failed to create environmental dict: {e}")
        return
    
    # Begin the navigation process
    # navigation(turtlebot_name, aruco_id, coordinates)
    # Begin the navigation process for all agents
    # run(agents, turtlebot_names, aruco_ids, coordinates)

    # 第一段导航任务
    # 获取第一段导航的agent参数，终点为aruco_waypoints_ids
    try:
        agent_yaml_params_dict_stage1 = create_agent_yaml_params(agents, aruco_ids, aruco_waypoints_ids)
    except Exception as e:
        rospy.logerr(f"Failed to create agent_yaml_params_dict_stage1: {e}")
        return
    rospy.loginfo(f"{agent_yaml_params_dict_stage1}\n")
    
    outputfile_path_stage1 = "./cbs_output_stage1.yaml"
    try:
        # Read and transform waypoints from the CBS output for stage 1
        coordinates_stage1 = read_and_transform_waypoints("./cbs_output.yaml", matrix, env_params, agent_yaml_params_dict_stage1, outputfile_path_stage1)
    except Exception as e:
        rospy.logerr(f"Failed to read and transform waypoints for stage 1: {e}")
        return

    # Begin the navigation process for all agents (第一段导航)
    rospy.loginfo("Starting stage 1 navigation...")
    run(agents, turtlebot_names, aruco_ids, coordinates_stage1, stage=1)

    # 第二段导航任务
    # 更新当前位姿的ArUco IDs（如果需要）
    # 如果机器人在导航过程中未改变ArUco定位的ID，则可以保持不变

    # 获取第二段导航的agent参数，终点为aruco_goalpoints_ids
    agent_yaml_params_dict_stage2 = create_agent_yaml_params(agents, aruco_ids, aruco_goalpoints_ids)
    outputfile_path_stage2 = "./cbs_output_stage2.yaml"

    try:
        # Read and transform waypoints from the CBS output for stage 2
        coordinates_stage2 = read_and_transform_waypoints("./cbs_output.yaml", matrix, env_params, agent_yaml_params_dict_stage2, outputfile_path_stage2)
    except Exception as e:
        rospy.logerr(f"Failed to read and transform waypoints for stage 2: {e}")
        return

    # Begin the navigation process for all agents (第二段导航)
    rospy.loginfo("Starting stage 2 navigation...")
    run(agents, turtlebot_names, aruco_ids, coordinates_stage2, stage=2)












if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

#!/usr/bin/env python3
"""
compare_bringup.launch.py

Brings up ONLY: map_server + planner_server (with SmacPlannerHybrid) + their
lifecycle manager. Deliberately does NOT bring up controller_server, bt_navigator,
or AMCL -- this is a global-planner-only comparison, not a full nav stack test.

Usage:
  ros2 launch compare_bringup.launch.py map_yaml:=/absolute/path/to/map.yaml

Run this in its own terminal, alongside (in a separate terminal) your own
a_star_node running as usual. Then run compare_planners.py in a third terminal.
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_params = os.path.join(pkg_dir, "config", "nav2_planner_params.yaml")

    map_yaml_arg = DeclareLaunchArgument(
        "map_yaml", description="Absolute path to map.yaml (the object_avoider map)"
    )
    params_arg = DeclareLaunchArgument(
        "params_file", default_value=default_params,
        description="Path to nav2_planner_params.yaml"
    )

    map_yaml = LaunchConfiguration("map_yaml")
    params_file = LaunchConfiguration("params_file")

    map_server_node = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[params_file, {"yaml_filename": map_yaml}],
    )

    planner_server_node = Node(
        package="nav2_planner",
        executable="planner_server",
        name="planner_server",
        output="screen",
        parameters=[params_file],
    )

    lifecycle_manager_node = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_planner",
        output="screen",
        parameters=[params_file],
    )

    return LaunchDescription([
        map_yaml_arg,
        params_arg,
        map_server_node,
        planner_server_node,
        lifecycle_manager_node,
    ])

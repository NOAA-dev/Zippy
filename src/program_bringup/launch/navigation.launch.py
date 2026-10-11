"""slam_toolbox (online async) + Nav2 servers. Final velocity goes to /autonomous_cmd_vel."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("program_bringup")
    nav2_params = os.path.join(pkg, "config", "nav2_params.yaml")
    slam_params = os.path.join(pkg, "config", "slam_params.yaml")

    use_sim_time = LaunchConfiguration("use_sim_time")
    sim_arg = DeclareLaunchArgument("use_sim_time", default_value="true")
    common = {"use_sim_time": use_sim_time}

    slam = Node(
        package="slam_toolbox", executable="sync_slam_toolbox_node", name="slam_toolbox",
        output="screen", parameters=[slam_params, common])

    controller = Node(
        package="nav2_controller", executable="controller_server", name="controller_server",
        output="screen", parameters=[nav2_params, common],
        remappings=[("cmd_vel", "cmd_vel_nav")])

    planner = Node(
        package="nav2_planner", executable="planner_server", name="planner_server",
        output="screen", parameters=[nav2_params, common])

    behaviors = Node(
        package="nav2_behaviors", executable="behavior_server", name="behavior_server",
        output="screen", parameters=[nav2_params, common],
        remappings=[("cmd_vel", "cmd_vel_nav")])

    bt = Node(
        package="nav2_bt_navigator", executable="bt_navigator", name="bt_navigator",
        output="screen", parameters=[nav2_params, common])

    smoother = Node(
        package="nav2_velocity_smoother", executable="velocity_smoother", name="velocity_smoother",
        output="screen", parameters=[nav2_params, common],
        remappings=[("cmd_vel", "cmd_vel_nav"), ("cmd_vel_smoothed", "autonomous_cmd_vel")])

    lifecycle = Node(
        package="nav2_lifecycle_manager", executable="lifecycle_manager", name="lifecycle_manager_navigation",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time, "autostart": True,
                     "node_names": ["controller_server", "planner_server", "behavior_server",
                                    "bt_navigator", "velocity_smoother"]}])
    slam_lifecycle = Node(
        package="nav2_lifecycle_manager", executable="lifecycle_manager", name="lifecycle_manager_slam", output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "autostart": True,
            "node_names": ["slam_toolbox"],
            "bond_timeout": 10.0,
            }],)

    return LaunchDescription([sim_arg, slam, controller, planner, behaviors, bt, smoother, lifecycle, slam_lifecycle])

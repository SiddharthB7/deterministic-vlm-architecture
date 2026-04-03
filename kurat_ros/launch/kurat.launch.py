from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    color_topic = LaunchConfiguration("color_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    query_topic = LaunchConfiguration("query_topic")
    reply_topic = LaunchConfiguration("reply_topic")
    enable_depth = LaunchConfiguration("enable_depth")
    log_level = LaunchConfiguration("log_level")
    stale_frame_threshold = LaunchConfiguration("stale_frame_threshold")
    ollama_host = LaunchConfiguration("ollama_host")

    return LaunchDescription(
        [
            DeclareLaunchArgument("color_topic", default_value="/camera/color/image_raw"),
            DeclareLaunchArgument("depth_topic", default_value="/camera/depth/image_rect_raw"),
            DeclareLaunchArgument("query_topic", default_value="/kurat/query"),
            DeclareLaunchArgument("reply_topic", default_value="/kurat/reply"),
            DeclareLaunchArgument("enable_depth", default_value="false"),
            DeclareLaunchArgument("log_level", default_value="INFO"),
            DeclareLaunchArgument("stale_frame_threshold", default_value="1.0"),
            DeclareLaunchArgument("ollama_host", default_value="http://127.0.0.1:11434"),
            Node(
                package="kurat",
                executable="kurat_orchestrator_node",
                name="kurat_orchestrator",
                output="screen",
                parameters=[
                    {
                        "color_topic": color_topic,
                        "depth_topic": depth_topic,
                        "query_topic": query_topic,
                        "reply_topic": reply_topic,
                        "enable_depth": enable_depth,
                        "log_level": log_level,
                        "stale_frame_threshold": stale_frame_threshold,
                        "ollama_host": ollama_host,
                    }
                ],
            ),
        ]
    )

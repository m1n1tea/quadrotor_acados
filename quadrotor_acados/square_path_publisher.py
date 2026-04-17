from typing import List, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node


class SquarePathPublisher(Node):
    def __init__(self) -> None:
        super().__init__("square_path_publisher")

        self.declare_parameter("path_topic", "/reference_path")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("side_length_m", 5.0)
        self.declare_parameter("altitude_m", -3.0)
        self.declare_parameter("points_per_edge", 25)
        self.declare_parameter("publish_rate_hz", 1.0)

        self.path_topic = str(self.get_parameter("path_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.side_length = float(self.get_parameter("side_length_m").value)
        self.altitude = float(self.get_parameter("altitude_m").value)
        self.points_per_edge = max(2, int(self.get_parameter("points_per_edge").value))
        self.publish_rate_hz = max(
            0.1, float(self.get_parameter("publish_rate_hz").value)
        )

        self.publisher = self.create_publisher(Path, self.path_topic, 10)
        self.path_msg = self._build_square_path()
        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self._publish_path)

        self.get_logger().info(
            f"Publishing square trajectory on {self.path_topic}: "
            f"side={self.side_length:.2f} m, altitude={self.altitude:.2f} m"
        )

    def _build_square_path(self) -> Path:
        corners: List[Tuple[float, float]] = [
            (0.0, 0.0),
            (self.side_length, 0.0),
            (self.side_length, self.side_length),
            (0.0, self.side_length),
            (0.0, 0.0),
        ]

        samples: List[Tuple[float, float]] = []
        for edge_idx in range(len(corners) - 2):
            x0, y0 = corners[edge_idx]
            x1, y1 = corners[edge_idx + 1]
            for i in range(self.points_per_edge):
                t = i / float(self.points_per_edge)
                x = (1.0 - t) * x0 + t * x1
                y = (1.0 - t) * y0 + t * y1
                samples.append((x, y))
        #samples.append(corners[-1])

        msg = Path()
        msg.header.frame_id = self.frame_id
        for x, y in samples:
            pose = PoseStamped()
            pose.header.frame_id = self.frame_id
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = float(self.altitude)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        return msg

    def _publish_path(self) -> None:
        now = self.get_clock().now().to_msg()
        self.path_msg.header.stamp = now
        for pose in self.path_msg.poses:
            pose.header.stamp = now
        self.publisher.publish(self.path_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SquarePathPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

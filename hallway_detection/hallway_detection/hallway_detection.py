"""
Publishes goal_poses for the nubot that will cause it to explore the environment. When the robot gets close to a wall, it will turn

PUBLISHERS:
  + binary_image (Image) - The binary image from the Deep Learning model

SUBSCRIBERS:
  + camera/XXXXX (Image) - The image from the camera on the robot

"""
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from enum import Enum, auto

class Hallway_Detection(Node):
    def __init__(self):
        """
        Initializes the Hallway_Detection node
        """
        super().__init__('hallway_detection')

        # Create tf listener
        self.buffer = Buffer()
        self.tf_listener = TransformListener(self.buffer, self)

        # Create goal_pose publisher
        self.goal_pose_publisher = self.create_publisher(PoseStamped, 'goal_pose', 10)

        # Create laser scan subscriber
        self.laser_scan_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.laser_scan_callback,
            10)
        
        # Create timer
        self.timer = self.create_timer(0.1, self.timer_callback)


    def timer_callback(self):
        pass

def main(args=None):
    """ The main() function. """
    rclpy.init(args=args)
    node = Hallway_Detection()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
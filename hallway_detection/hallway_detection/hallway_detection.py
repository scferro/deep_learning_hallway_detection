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
from sensor_msgs.msg import Image
from enum import Enum, auto

class Hallway_Detection(Node):
    def __init__(self):
        """
        Initializes the Hallway_Detection node
        """
        super().__init__('hallway_detection')

        self.declare_parameter("rate", 5)
        rate = self.get_parameter('rate').get_parameter_value().double_value

        # Create goal_pose publisher
        self.image_out_pub = self.create_publisher(Image, 'binary_image', 10)

        # Create laser scan subscriber
        self.image_sub = self.create_subscription(
            Image,
            'scan',
            self.image_callback,
            10)
        
        # Create timer
        rate_seconds = 1 / rate
        self.timer = self.create_timer(rate_seconds, self.timer_callback)

    def timer_callback(self):
        pass

    def image_callback(self):
        pass

def main(args=None):
    """ The main() function. """
    rclpy.init(args=args)
    node = Hallway_Detection()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
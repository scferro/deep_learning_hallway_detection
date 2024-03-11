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
from sensor_msgs.msg import Image
import torch
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class Hallway_Detection(Node):
    def __init__(self):
        """
        Initializes the Hallway_Detection node
        """
        super().__init__('hallway_detection')

        self.declare_parameter("rate", 5.)
        self.rate = self.get_parameter('rate').get_parameter_value().double_value
        
        self.current_image = Image

        # Create goal_pose publisher
        self.image_out_pub = self.create_publisher(Image, 'binary_image', 10)

        # Create laser scan subscriber
        self.image_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10)
        
        # Create timer
        rate_seconds = 1 / self.rate
        self.timer = self.create_timer(rate_seconds, self.timer_callback)

    def timer_callback(self):
        pass

    def image_callback(self, image):
        self.current_image = image

    def ros2_image_to_tensor(ros_image: Image) -> torch.Tensor:
        """
        Converts a ROS2 image message to a PyTorch tensor with normalized values (0 to 1).

        :param ros_image: The ROS2 image message to be converted.
        :return: A PyTorch tensor representing the image, with pixel values normalized to [0, 1].
        """
        bridge = CvBridge()
        
        # Convert ROS2 image to OpenCV image
        try:
            cv_image = bridge.imgmsg_to_cv2(ros_image, desired_encoding='passthrough')
        except Exception as e:
            print(f"Error converting ROS2 image to OpenCV: {e}")
            return None

        # Convert the OpenCV image to a PyTorch tensor and normalize it
        # Note: cv_image might need to be converted from BGR to RGB if you're working with color images,
        # depending on your application's requirements.
        image_tensor = torch.from_numpy(cv_image).float()
        
        # Normalize the tensor to [0, 1] by dividing by the max value (255 for 8-bit images)
        if len(image_tensor.shape) == 3:  # For color images
            image_tensor = image_tensor.permute(2, 0, 1)  # Move the channel to the first dimension
        image_tensor /= 255.0

        return image_tensor

def main(args=None):
    """ The main() function. """
    rclpy.init(args=args)
    node = Hallway_Detection()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
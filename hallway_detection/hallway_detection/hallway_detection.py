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
import cv2
# from sensor_msgs.msg import Image


class Autoencoder(torch.nn.Module):

  def __init__(self, n_classes):
    super(Autoencoder, self).__init__()
    # encoder layers
    self.enc_conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding='same')
    self.enc_conv1_2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same')
    self.enc_bn1 = torch.nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True)
    self.enc_pool1 = torch.nn.MaxPool2d((2,2), stride=2)
    self.enc_relu1 = torch.nn.ReLU()
    self.enc_conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same')
    self.enc_conv2_2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same')
    self.enc_bn2 = torch.nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True)
    self.enc_pool2 = torch.nn.MaxPool2d((2,2), stride=2)
    self.enc_relu2 = torch.nn.ReLU()
    self.enc_conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding='same')
    self.enc_conv3_2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding='same')
    self.enc_bn3 = torch.nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True)
    self.enc_pool3 = torch.nn.MaxPool2d((2,2), stride=2)
    self.enc_relu3 = torch.nn.ReLU()
    self.enc_conv4 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding='same')
    self.enc_conv4_2 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding='same')
    self.enc_bn4 = torch.nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True)
    self.enc_pool4 = torch.nn.MaxPool2d((2,2), stride=2)
    self.enc_relu4 = torch.nn.ReLU()
    # decoder layers
    self.dec_up1 = torch.nn.UpsamplingNearest2d(scale_factor=2)
    self.dec_conv1 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding='same')
    self.dec_conv1_2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding='same')
    self.dec_bn1 = torch.nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True)
    self.dec_relu1 = torch.nn.ReLU()
    self.dec_up2 = torch.nn.UpsamplingNearest2d(scale_factor=2)
    self.dec_conv2 = torch.nn.Conv2d(in_channels=128, out_channels=32, kernel_size=5, stride=1, padding='same')
    self.dec_conv2_2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same')
    self.dec_bn2 = torch.nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True)
    self.dec_relu2 = torch.nn.ReLU()
    self.dec_up3 = torch.nn.UpsamplingNearest2d(scale_factor=2)
    self.dec_conv3 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding='same')
    self.dec_conv3_2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same')
    self.dec_bn3 = torch.nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True)
    self.dec_relu3 = torch.nn.ReLU()
    self.dec_up4 = torch.nn.UpsamplingNearest2d(scale_factor=2)
    self.dec_conv4 = torch.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, stride=1, padding='same')
    self.dec_conv4_2 = torch.nn.Conv2d(in_channels=3, out_channels=n_classes, kernel_size=5, stride=1, padding='same')
    self.dec_bn4 = torch.nn.BatchNorm2d(num_features=n_classes, eps=1e-05, momentum=0.1, affine=True)
    self.dec_relu4 = torch.nn.ReLU()
    #self.dec_soft = torch.nn.Softmax(dim=1)

  def forward(self, x):
    # encoder forward pass
    a = self.enc_conv1(x)
    a = self.enc_conv1_2(a)
    a = self.enc_bn1(a)
    a = self.enc_relu1(a)
    res1 = self.enc_pool1(a)

    a = self.enc_conv2(res1)
    a = self.enc_conv2_2(a)
    a = self.enc_bn2(a)
    a = self.enc_relu2(a)
    res2 = self.enc_pool2(a)

    a = self.enc_conv3(res2)
    a = self.enc_conv3_2(a)
    a = self.enc_bn3(a)
    a = self.enc_relu3(a)
    res3 = self.enc_pool3(a)

    a = self.enc_conv4(res3)
    a = self.enc_conv4_2(a)
    a = self.enc_bn4(a)
    a = self.enc_relu4(a)
    res4 = self.enc_pool4(a)

    # decoder forward pass
    a = self.dec_up1(res4)
    a = self.dec_conv1(a)
    a = self.dec_conv1_2(a)
    a = self.dec_bn1(a)
    a = self.dec_relu1(a)

    a = torch.cat((a,res3),1)

    a = self.dec_up2(a)
    a = self.dec_conv2(a)
    a = self.dec_conv2_2(a)
    a = self.dec_bn2(a)
    a = self.dec_relu2(a)

    a = torch.cat((a,res2),1)

    a = self.dec_up3(a)
    a = self.dec_conv3(a)
    a = self.dec_conv3_2(a)
    a = self.dec_bn3(a)
    a = self.dec_relu3(a)

    a = torch.cat((a,res1),1)

    a = self.dec_up4(a)
    a = self.dec_conv4(a)
    a = self.dec_conv4_2(a)
    a = self.dec_bn4(a)
    a = self.dec_relu4(a)

    return a #self.dec_soft(a)


class Hallway_Detection(Node):
    def __init__(self):
        """
        Initializes the Hallway_Detection node
        """
        super().__init__('hallway_detection')

        self.declare_parameter("rate", 30.)
        self.rate = self.get_parameter('rate').get_parameter_value().double_value
        self.current_image = None
        # Create goal_pose publisher
        self.image_out_pub = self.create_publisher(Image, 'binary_image', 10)
        self.red_green_pub = self.create_publisher(Image, 'red_green', 10)
        
        
        # Create laser scan subscriber
        self.image_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10)
        
        # Create timer
        rate_seconds = 1 / self.rate
        self.timer = self.create_timer(rate_seconds, self.timer_callback)
        self.model = Autoencoder(2)
        self.model.load_state_dict(torch.load('/home/sdalal/ws/deep_learning_final/model'))
    

    def timer_callback(self):
        if self.current_image is not None:

            image_tensor = self.ros2_image_to_tensor(self.current_image)
            output_tensor = self.model_inference(image_tensor, self.model)[0].argmax(dim=0)
            # output_image = self.tensor_to_mask_image(output_tensor)

            output_image = self.tensor_to_ros2_image(output_tensor)
            
            # red_green_image = self.apply_binary_filter_to_image(self.current_image, output_image)
            # self.get_logger().error(output_image)
            self.image_out_pub.publish(output_image)
            # self.red_green_pub.publish(red_green_image)
        

    def image_callback(self, image):
 
        self.current_image = image
        # image_tensor = self.ros2_image_to_tensor(self.current_image)
        # print(image_tensor)

    
    def ros2_image_to_tensor(self, ros_image: Image) -> torch.Tensor:
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
            self.get_logger().error(f"Error converting ROS2 image to OpenCV: {e}")
            return None
        
        #crop the image to 480 x 480
        cv_image = cv_image[0:720, 180:900]
        cv_image = cv2.resize(cv_image, (480, 480))


        # Convert the OpenCV image to a PyTorch tensor and normalize it
        # Note: cv_image might need to be converted from BGR to RGB if you're working with color images,
        # depending on your application's requirements.
        image_tensor = torch.from_numpy(np.array([cv_image])).float()
        
        # Normalize the tensor to [0, 1] by dividing by the max value (255 for 8-bit images)
        if len(image_tensor.shape) == 4:  # For color images
            image_tensor = image_tensor.permute(0, 3, 1, 2)  # Move the channel to the first dimension
        image_tensor /= 255.0

        return image_tensor
    
    # def apply_binary_filter_to_image(self,image, filter_image):

    #     # Ensure the filter is binary
    #     bridge = CvBridge()
    #     try:
    #         image = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
    #         filter_image = bridge.imgmsg_to_cv2(filter_image, desired_encoding='passthrough')
    #     except Exception as e:
    #         self.get_logger().error(f"Error converting ROS2 image to OpenCV: {e}")
    #         return None
        
    #     filter_image = np.where(np.array(filter_image) > 127, 255, 0).astype(np.uint8)
    #     # filter_image = Image.fromarray(filter_image)
    #     #copnvert the filter image to a ros image
    #     filter_image = bridge.cv2_to_imgmsg(filter_image, encoding='mono8')
        
    #     x = np.zeros((480,480,3))
    #     y = np.array(filter_image)
    #     np.where(y ==0.0)
    #     i,j = np.where(y!=0.0)

    #     for a in range(len(i)):
    #         x[i[a],j[a],0] = 255

    #     i,j = np.where(y==0.0)

    #     for a in range(len(i)):
    #         x[i[a],j[a],1] = 255
    #     filter_img = Image.fromarray(x.astype('uint8'))
    #     output_image = Image.blend(image, filter_img, 0.25)

    #     return output_image

    
    #convert tensor to ros2 image
    def tensor_to_ros2_image(self, tensor: torch.Tensor) -> Image:
        """
        Converts a PyTorch tensor to a ROS2 Image message.

        :param tensor: The PyTorch tensor to be converted.
        :return: A ROS2 Image message representing the tensor.
        """
        bridge = CvBridge()

        # Convert the PyTorch tensor to a NumPy array
        
        tensor = tensor * 255.0
        image_np = tensor.detach().numpy().astype(np.uint8)
        
        # self.get_logger().error(str(image_np.shape))
        image_np = image_np.reshape((480, 480, 1))    
        

        # Convert the NumPy array to a ROS2 Image message
        try:
            image_msg = bridge.cv2_to_imgmsg(image_np, encoding='mono8')
        except Exception as e:
            print(f"Error converting NumPy array to ROS2 Image: {e}")
            return None

        return image_msg

    # Use a trained model to take in an image tensor and convert the image to the output of the model
    def model_inference(self, image_tensor: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        """
        Uses a trained model to perform inference on an image tensor.

        :param image_tensor: The input image tensor to be processed.
        :param model: The PyTorch model to be used for inference.
        :return: The output tensor from the model.
        """
        # Set the model to evaluation mode
        # model.eval()

        # Perform inference
        # self.get_logger().error(str(image_tensor))
        output = model(image_tensor)

        return output
        
    def tensor_to_mask_image(self, tensor: torch.Tensor) -> Image:
        """
        Converts a PyTorch tensor to a ROS2 Image message.

        :param tensor: The PyTorch tensor to be converted.
        :return: A ROS2 Image message representing the tensor.
        """
        bridge = CvBridge()

        # Convert the PyTorch tensor to a NumPy array
        image_np = tensor.numpy()

        # Convert the NumPy array to a ROS2 Image message
        try:
            image_msg = bridge.cv2_to_imgmsg(image_np, encoding='passthrough')
        except Exception as e:
            print(f"Error converting NumPy array to ROS2 Image: {e}")
            return None

        return image_msg

def main(args=None):
    """ The main() function. """
    rclpy.init(args=args)
    node = Hallway_Detection()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
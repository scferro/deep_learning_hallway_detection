# Obstacle Detection from RGB Video Using Deep Learning

## Authors:
Stephen Ferro, Max Palay, Shail Dalal

## Description
This is a collection of ROS 2 packages for deploying a trained obstacle deetction nueral network on a obile robot with an Intel RealSense camera.

hallway_detection - a package for detecting walls and obstales while navigating through hallways. 

## Launchfiles

Use `detect_hallways.launch.xml` to launch the `hallway_detection` node. This node subscribes to the `/camera/color/image_raw` topic publihsed by the RealSense camera, and publishes binary segmented images showing safe regions for the robot to travel to on the `/binary_image` topic.

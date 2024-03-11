from setuptools import find_packages, setup

package_name = 'hallway_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
                                   'launch/detect_hallways.launch.xml',
                                   'config/hallway_detection.rviz',
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='scferro',
    maintainer_email='stephencferro@gmail.com',
    description='A package for controlling an Ackermann robot using feedback from a camera mounted on the robot.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hallway_detection = hallway_detection.hallway_detection:main'
        ],
    },
)

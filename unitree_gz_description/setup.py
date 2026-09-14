import os
from glob import glob

from setuptools import setup

package_name = 'unitree_gz_description'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'robots'), glob('robots/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kieran Paranjpe',
    maintainer_email='kieran.paranjpe@mail.mcgill.ca',
    description='Generic bare-URDF to Gazebo/ros2_control URDF generator.',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [],
    },
)

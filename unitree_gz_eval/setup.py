from glob import glob

from setuptools import setup

package_name = 'unitree_gz_eval'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/suites', glob('suites/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kieran Paranjpe',
    maintainer_email='kieran.paranjpe@mail.mcgill.ca',
    description='Fixed Gazebo test suite for any velocity policy, graded on simulator ground truth.',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'eval_runner = unitree_gz_eval.runner_node:main',
        ],
    },
)

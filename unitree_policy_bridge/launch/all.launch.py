# Everything in one command: the single-node bridge, then the simulator.
from unitree_gz_bringup.launch_utils import combined_launch_description


def generate_launch_description():
    return combined_launch_description('unitree_policy_bridge', 'bridge.launch.py')

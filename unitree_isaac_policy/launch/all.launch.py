# Everything in one command: policy_node + pd_node, then the simulator.
from unitree_gz_bringup.launch_utils import combined_launch_description


def generate_launch_description():
    return combined_launch_description('unitree_isaac_policy', 'policy.launch.py')

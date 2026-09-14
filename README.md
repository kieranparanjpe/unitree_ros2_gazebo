# unitree_ros2_gazebo

Sim2sim deployment of Isaac Lab-trained locomotion policies into ROS 2 + Gazebo.

An Isaac Lab training run exports a `deploy.yaml` manifest describing its observation and
action contract (term order, scales, gains, default pose, joint mapping). The three packages
here read that manifest and run the exported ONNX policy against a Gazebo-simulated robot,
with no per-robot code. Currently exercised with Unitree H2.

| Package | Language | Role |
| --- | --- | --- |
| `unitree_gz_description` | Python | Turns a bare URDF + `deploy.yaml` into a Gazebo/`ros2_control`-ready URDF and controller config |
| `unitree_gz_bringup` | Python | Launch file, world, and `gz_reset_node` (places the robot in the trained start pose) |
| `unitree_policy_bridge` | C++ | Parses `deploy.yaml`, runs the ONNX policy, converts its actions to joint torques |

## Requirements

Targets **ROS 2 Jazzy** and **Gazebo Harmonic** (`gz sim` 8.x) on Ubuntu 24.04. Gazebo
Classic and `gazebo_ros2_control` are *not* supported — the `/gazebo/*` services you'll find
in older tutorials do not exist in this stack.

```bash
# ROS 2 Jazzy - follow https://docs.ros.org/en/jazzy/Installation.html, then:
sudo apt install ros-jazzy-desktop

# Gazebo Harmonic + the ROS integration, ros2_control, and the controllers used here
sudo apt install \
  ros-jazzy-ros-gz-sim ros-jazzy-ros-gz-bridge \
  ros-jazzy-gz-ros2-control \
  ros-jazzy-ros2-control ros-jazzy-ros2-controllers ros-jazzy-controller-manager \
  ros-jazzy-forward-command-controller ros-jazzy-joint-state-broadcaster \
  ros-jazzy-robot-state-publisher
```

Verify Gazebo is the right generation:

```bash
gz sim --version      # expect 8.x
```

You also need, from outside this repo:

- **A bare robot URDF plus its `meshes/` directory.** For H2, `unitree_ros/robots/h2_description/H2.urdf`. Meshes are resolved relative to the URDF's own directory, so keep the folder intact.
- **A trained policy export**, laid out in Isaac Lab's own convention:
  ```
  <policy_dir>/exported/policy.onnx
  <policy_dir>/params/deploy.yaml
  ```
  You pass the path to `policy.onnx`; `deploy.yaml` is found as the sibling `params/`
  directory. Nothing else about the training run is needed.
- **ONNX Runtime** (Linux x64 release) — see below.

## ONNX Runtime

`unitree_policy_bridge` links against a local ONNX Runtime release. There is no auto-detection
and no environment variable, so point it at your copy one of two ways.

Either edit the placeholder near the top of `unitree_policy_bridge/CMakeLists.txt`:

```cmake
if(NOT DEFINED ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_ROOT "/path/to/onnxruntime-linux-x64-1.22.0")
endif()
```

or leave the file alone and override it at build time:

```bash
colcon build --cmake-args -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.22.0
```

The directory must contain `include/onnxruntime_cxx_api.h` and `lib/libonnxruntime.so*`; the
build fails with an explicit message if it doesn't. If you already have `unitree_rl_lab`
checked out, it vendors a suitable copy under
`unitree_rl_lab/deploy/thirdparty/onnxruntime-linux-x64-1.22.0/`.

## Build

The three packages need to be in a colcon workspace. The workspace is deliberately kept
outside this repo — symlinking in is the intended pattern, so the repo stays a plain
submodule:

```bash
mkdir -p ~/jazzy_ws/src
ln -s /path/to/unitree_ros2_gazebo/unitree_gz_description ~/jazzy_ws/src/
ln -s /path/to/unitree_ros2_gazebo/unitree_gz_bringup    ~/jazzy_ws/src/
ln -s /path/to/unitree_ros2_gazebo/unitree_policy_bridge ~/jazzy_ws/src/

source /opt/ros/jazzy/setup.bash
cd ~/jazzy_ws
colcon build --symlink-install --packages-select \
  unitree_gz_description unitree_gz_bringup unitree_policy_bridge
```

## Run

```bash
source /opt/ros/jazzy/setup.bash
source ~/jazzy_ws/install/setup.bash

ros2 launch unitree_gz_bringup sim.launch.py \
  policy_onnx_path:=/path/to/<policy_dir>/exported/policy.onnx \
  urdf_path:=/path/to/unitree_ros/robots/h2_description/H2.urdf
```

Launch arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `policy_onnx_path` | *(required)* | Exported `policy.onnx`; `deploy.yaml` is derived as the sibling `params/` directory |
| `urdf_path` | *(required)* | Bare robot URDF, with `meshes/` alongside it |
| `robot` | `h2` | Selects `unitree_gz_description/robots/<name>.yaml` |

Everything is an explicit path — nothing assumes where your URDF or training runs live.

### What you should see

1. Gazebo opens **paused**, and the robot appears already in the trained crouch (not a
   straight-legged zero pose). It should be motionless.
2. A fraction of a second of simulation is stepped through so the controllers can activate —
   activation is serviced by the `controller_manager` update loop, so it needs steps.
3. `gz_reset_node` places the robot in `deploy.yaml`'s exact start state and the world
   un-pauses. The log reads:
   ```
   Startup reset complete (31 joints reset to default pose) after 0.070s of simulation
   Reset notification received - re-seeding on post-reset data.
   Starting the trained policy from the reset default pose.
   ```
4. The policy takes over immediately and the robot stands. With no velocity command it holds
   position; `ang_vel_b` should stay well under ~0.5 rad/s and `proj_grav_b` near `[0,0,-1]`.

### Driving the robot

The policy takes a velocity command on `/cmd_vel`. With nothing publishing, the command is
`[0,0,0]`, which the policy interprets as "stand still" — so a robot that stands but doesn't
walk is working correctly, not stuck.

```bash
# walk forward at 0.5 m/s
ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5}}"

# turn
ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist "{angular: {z: 0.2}}"

# arc
ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.4}, angular: {z: 0.2}}"
```

Commands are clamped to the ranges the policy was trained on, read from `deploy.yaml`. For
the current H2 checkpoint:

| Field | Trained range |
| --- | --- |
| `linear.x` | -0.5 … 1.0 m/s |
| `linear.y` | -0.3 … 0.3 m/s |
| `angular.z` | -0.2 … 0.2 rad/s |

Anything outside is silently clamped. Stop publishing and the robot returns to standing.

Measured on the H2 checkpoint: forward tracking is good (~0.43 m/s achieved for 0.5
commanded), yaw tracking is poor (~0.03 rad/s achieved for 0.2 commanded). The yaw weakness
appears to be a property of this training run rather than the bridge — the command reaches
the observation vector correctly, which the `cmd=[...]` field of the diagnostic log confirms.

### Resetting mid-run

`gz_reset_node` also exposes the reset as a service, which returns the robot to the trained
start state without restarting Gazebo:

```bash
ros2 service call /gz_reset_node/reset std_srvs/srv/Trigger
```

This restores joint positions, joint velocities, base pose and base velocity, and tells the
policy bridge to re-seed its observation history.

## Adding another robot

For a robot whose `deploy.yaml` uses the same observation and action terms as H2
(`base_ang_vel`, `projected_gravity`, `velocity_commands`, `joint_pos_rel`, `joint_vel_rel`,
`last_action`, and a single `JointPositionAction`), add one file —
`unitree_gz_description/robots/<name>.yaml`:

```yaml
imu_link: <link the IMU attaches to>
spawn_height: <base height of the standing pose, metres>
joint_sdk_names:      # the robot's SDK motor order, one name per motor
  - ...
```

Then launch with `robot:=<name>` and the new `urdf_path`/`policy_onnx_path`. No code changes.
A new *observation term type* is the one thing that does need code: a branch in
`observation_term_fn()` in `unitree_policy_bridge/src/policy_bridge_node.cpp`.

## Notes and gotchas

- **`deploy.yaml` is not uniformly ordered.** `stiffness` and `damping` are exported in the
  robot's SDK motor order, while `default_joint_pos`, the action scale/offset/clip and the
  observation scales are in Isaac's environment joint order. Reading them all positionally
  silently gives each joint another joint's gains. `load_dynamics()` re-indexes them through
  `joint_ids_map`. This is worth knowing before trusting any other array in that file.
- **The control loop runs on simulation time.** The launch bridges `/clock` and every node
  sets `use_sim_time`. Without the clock bridge the policy would step on wall time and run at
  a rate unrelated to how fast physics is advancing.
- **The start pose comes from `initial_value`.** `generate_urdf.py` declares each joint's
  `default_joint_pos` as a nested `<param name="initial_value">` on its position state
  interface, which `gz_ros2_control` applies as a joint teleport at spawn. It must be a
  nested element — `ros2_control` silently ignores it as an attribute.
- **The world file is custom** (`unitree_gz_bringup/worlds/default.sdf`). Gazebo's stock
  `empty.sdf` doesn't load `gz-sim-imu-system`, so an IMU sensor never produces data on it.
- **There is no stand-up phase.** The reset establishes the trained start state directly, so
  the policy runs from the first step, as it does in Isaac Lab. On real hardware, where
  nothing can teleport the robot, reaching the default pose is the controller FSM's job
  upstream of this node; `wait_for_reset` (a `unitree_policy_bridge` parameter, default
  `false`) is what makes the simulation wait for the teleport instead.

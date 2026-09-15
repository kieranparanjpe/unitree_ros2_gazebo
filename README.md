# unitree_ros2_gazebo

Sim2sim deployment of Isaac Lab-trained locomotion policies into ROS 2 + Gazebo.

An Isaac Lab training run exports a `deploy.yaml` manifest describing its observation and
action contract (term order, scales, gains, default pose, joint mapping). The packages here
read that manifest and run the exported ONNX policy against a Gazebo-simulated robot, with no
per-robot code. Currently exercised with Unitree H2.

There are **two interchangeable policy stacks**. They do the same job and are mutually
exclusive — both drive `/effort_controller/commands`, so run one or the other, never both.

| | `unitree_isaac_policy` | `unitree_policy_bridge` |
| --- | --- | --- |
| Structure | Two nodes: inference, then PD | One node doing both |
| Observation/action pipeline | **Reuses `unitree_rl_lab/deploy` verbatim** | Re-implemented against `deploy.yaml` |
| Observation terms supported | Whatever upstream registers (incl. `gait_phase`, mimic terms) | The six H2 uses, via an if-else chain |
| Extra build dependency | `unitree_rl_lab` + `unitree_sdk2` headers | none |

`unitree_isaac_policy` is the one to prefer for new work: its observation vector is built by
the same code Unitree's own DDS hardware / `unitree_mujoco` deploy path runs, so parity is
structural rather than something to test. `unitree_policy_bridge` is self-contained and needs
no `unitree_rl_lab` checkout.

## Packages

| Package | Language | Role |
| --- | --- | --- |
| `unitree_gz_description` | Python | Library, no nodes. Turns a bare URDF + `deploy.yaml` into a Gazebo/`ros2_control`-ready URDF and controller config |
| `unitree_gz_bringup` | Python | The simulator: launch file, world, and `gz_reset_node` |
| `unitree_isaac_policy` | C++ | Policy stack A — `policy_node` + `pd_node` |
| `unitree_policy_bridge` | C++ | Policy stack B — `policy_bridge_node` |

### `unitree_gz_description`

No nodes; it is imported by the launch files. `generate_urdf.py` injects the `<ros2_control>`
block (one joint per motor, each carrying its `default_joint_pos` as a nested
`<param name="initial_value">`), the `gz_ros2_control` plugin, and an IMU sensor, and builds
the `controller_manager` YAML. `robots/<name>.yaml` holds the only per-robot data that is
neither a path nor derivable: `imu_link`, `joint_sdk_names`, `spawn_height`.

It depends on nothing but `python3-yaml` — `generate_urdf.py` imports only `os`, `xml.etree`
and `yaml`. That is deliberate: both policy packages import it for the joint-order derivation,
and on the hardware path there is no Gazebo to depend on.

### `unitree_gz_bringup`

`launch_utils.py` composes a policy launch file with the simulator launch in the correct order
(see `all.launch.py` below) — it lives here rather than in `unitree_gz_description` because it
starts `sim.launch.py`, and having it on the other side made the two packages mutually
dependent.

`gz_reset_node` places the robot in `deploy.yaml`'s exact start state — joint positions, joint
velocities, base pose and base velocity — by writing the Gazebo ECM directly, then un-pauses
the world and publishes a latched `/policy_reset`. It does this once at startup, and again on
demand via its `~/reset` service.

### `unitree_isaac_policy`

| Node | Subscribes | Publishes | Job |
| --- | --- | --- | --- |
| `policy_node` | `/joint_states`, `/imu`, `/cmd_vel`, `/policy_reset` | `/joint_targets` | Builds the observation vector and runs the ONNX policy at `deploy.yaml`'s `step_dt` (50 Hz for H2) |
| `pd_node` | `/joint_states`, `/joint_targets` | `/effort_controller/commands` | `tau = kp*(q_des - q) - kd*dq` at 1000 Hz |

`policy_node` reuses `ManagerBasedRLEnv`, `ObservationManager`, `ActionManager`, `OrtRunner`
and the mdp term registries from `unitree_rl_lab/deploy` unchanged. Only two things are
substituted: a `RosArticulation` feeding robot state from ROS topics instead of a DDS
`LowState`, and a `velocity_commands` term reading `/cmd_vel` instead of the handheld joystick
(installed by reassigning that entry in upstream's `observations_map()`).

The split is deliberate. Every other deploy target — real hardware, `unitree_mujoco`, Isaac
itself — has something downstream that converts a position target to torque (motor firmware,
MuJoCo's actuator model, Isaac's implicit-PD actuator). Gazebo's effort interface does not, so
`pd_node` is the only genuinely new code in the stack. Its math lives in
`include/unitree_isaac_policy/pd.hpp` with no ROS dependency, so a `ros2_control` controller
plugin could reuse it later without a rewrite.

### `unitree_policy_bridge`

`policy_bridge_node` does all of the above in one process: parses `deploy.yaml`, builds the
observation vector from its own term registry, runs the policy on a 50 Hz sim-time timer, and
runs the PD on a separate 1000 Hz timer.

## Requirements

Targets **ROS 2 Jazzy** and **Gazebo Harmonic** (`gz sim` 8.x) on Ubuntu 24.04. Gazebo Classic
and `gazebo_ros2_control` are *not* supported — the `/gazebo/*` services you'll find in older
tutorials do not exist in this stack.

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

# C++ libraries. yaml-cpp is used by both policy packages; eigen/spdlog/fmt by
# unitree_isaac_policy only (the unitree_rl_lab headers it reuses pull them in).
sudo apt install libyaml-cpp-dev libeigen3-dev libspdlog-dev libfmt-dev
```

Verify Gazebo is the right generation:

```bash
gz sim --version      # expect 8.x
```

You also need, from outside this repo:

- **A bare robot URDF plus its `meshes/` directory.** For H2,
  `unitree_ros/robots/h2_description/H2.urdf`. Meshes are resolved relative to the URDF's own
  directory, so keep the folder intact.
- **A trained policy export**, laid out in Isaac Lab's own convention:
  ```
  <policy_dir>/exported/policy.onnx
  <policy_dir>/params/deploy.yaml
  ```
  You pass the path to `policy.onnx`; `deploy.yaml` is found as the sibling `params/`
  directory. Nothing else about the training run is needed.
- **ONNX Runtime** (Linux x64 release).
- **A `unitree_rl_lab` and a `unitree_sdk2` checkout** — for `unitree_isaac_policy` only.

## Build-time paths

Neither policy package assumes this repo is checked out next to anything, so external paths
are explicit CMake cache variables with an editable placeholder near the top of the relevant
`CMakeLists.txt`. Either edit the placeholder or pass `-D<VAR>=<path>` at build time. Each is
checked, and the build fails with an explicit message if the path is wrong.

| Variable | Needed by | Points at |
| --- | --- | --- |
| `ONNXRUNTIME_ROOT` | both | An `onnxruntime-linux-x64-*` release containing `include/onnxruntime_cxx_api.h` and `lib/libonnxruntime.so*` |
| `UNITREE_RL_LAB_ROOT` | `unitree_isaac_policy` | A `unitree_rl_lab` checkout; its `deploy/include` supplies the reused runtime |
| `UNITREE_SDK2_ROOT` | `unitree_isaac_policy` | A `unitree_sdk2` checkout, for `unitree_joystick.hpp` (which despite its `dds_wrapper` path is standalone and pulls in no DDS) |

```bash
colcon build --cmake-args \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.22.0 \
  -DUNITREE_RL_LAB_ROOT=/path/to/unitree_rl_lab \
  -DUNITREE_SDK2_ROOT=/path/to/unitree_sdk2
```

`unitree_rl_lab` vendors a suitable ONNX Runtime under
`deploy/thirdparty/onnxruntime-linux-x64-1.22.0/`, which is what `UNITREE_RL_LAB_ROOT` makes
`ONNXRUNTIME_ROOT` default to in `unitree_isaac_policy`.

## Build

The packages need to be in a colcon workspace. The workspace is deliberately kept outside this
repo — symlinking in is the intended pattern, so the repo stays a plain submodule. Symlink the
whole repo and colcon finds the packages underneath it:

```bash
mkdir -p ~/jazzy_ws/src
ln -s /path/to/unitree_ros2_gazebo ~/jazzy_ws/src/

source /opt/ros/jazzy/setup.bash
cd ~/jazzy_ws
colcon build --symlink-install
```

Symlinking the individual package directories instead also works — but do one or the other,
not both, or colcon will see each package twice.

To skip `unitree_isaac_policy`, and with it the `unitree_rl_lab`/`unitree_sdk2` dependency:

```bash
colcon build --symlink-install --packages-select \
  unitree_gz_description unitree_gz_bringup unitree_policy_bridge
```

## Launch files

| Launch file | Starts | Use it for |
| --- | --- | --- |
| `unitree_isaac_policy all.launch.py` | `policy_node` + `pd_node`, then the simulator | Normal use, stack A |
| `unitree_policy_bridge all.launch.py` | `policy_bridge_node`, then the simulator | Normal use, stack B |
| `unitree_isaac_policy policy.launch.py` | `policy_node` + `pd_node` only | Iterating on a policy against a running sim |
| `unitree_policy_bridge bridge.launch.py` | `policy_bridge_node` only | Same, stack B |
| `unitree_gz_bringup sim.launch.py` | Gazebo, the robot, controllers, `gz_reset_node` | The simulator on its own |

Both `all.launch.py` files are a few lines delegating to `combined_launch_description()` in
`unitree_gz_bringup/launch_utils.py`, which includes the matching policy launch file and
then `sim.launch.py`.

**Ordering is not cosmetic.** The policy stack must be commanding effort before the world
un-pauses. `gz_reset_node` un-pauses as soon as its startup reset lands, and a robot that is
free-falling by then is not reliably recovered by a later teleport. `all.launch.py` handles
this by starting the policy first and delaying the simulator by `sim_delay` seconds.

## Run

```bash
source /opt/ros/jazzy/setup.bash
source ~/jazzy_ws/install/setup.bash
```

### A. `unitree_isaac_policy` — reuses the upstream deploy runtime

One command:

```bash
ros2 launch unitree_isaac_policy all.launch.py \
  policy_onnx_path:=/path/to/<policy_dir>/exported/policy.onnx \
  urdf_path:=/path/to/unitree_ros/robots/h2_description/H2.urdf
```

Or split across two terminals, to restart the policy without restarting Gazebo. Policy first;
once both are up, re-seed with the reset service (below).

```bash
# terminal 1
ros2 launch unitree_isaac_policy policy.launch.py \
  policy_onnx_path:=/path/to/<policy_dir>/exported/policy.onnx

# terminal 2
ros2 launch unitree_gz_bringup sim.launch.py \
  deploy_yaml_path:=/path/to/<policy_dir>/params/deploy.yaml \
  urdf_path:=/path/to/unitree_ros/robots/h2_description/H2.urdf
```

### B. `unitree_policy_bridge` — the self-contained single node

Identical arguments; substitute the package name.

```bash
ros2 launch unitree_policy_bridge all.launch.py \
  policy_onnx_path:=/path/to/<policy_dir>/exported/policy.onnx \
  urdf_path:=/path/to/unitree_ros/robots/h2_description/H2.urdf
```

```bash
# split form
ros2 launch unitree_policy_bridge bridge.launch.py policy_onnx_path:=...
ros2 launch unitree_gz_bringup sim.launch.py deploy_yaml_path:=... urdf_path:=...
```

### Arguments

`all.launch.py` (both packages):

| Argument | Default | Meaning |
| --- | --- | --- |
| `policy_onnx_path` | *(required)* | Exported `policy.onnx`; `deploy.yaml` is derived as the sibling `params/` directory |
| `urdf_path` | *(required)* | Bare robot URDF, with `meshes/` alongside it |
| `robot` | `h2` | Selects `unitree_gz_description/robots/<name>.yaml` |
| `sim_delay` | `5.0` | Seconds to let the policy stack load before the simulator starts |

`sim.launch.py`:

| Argument | Default | Meaning |
| --- | --- | --- |
| `deploy_yaml_path` | *(required)* | Needed for the joint order and default pose that shape the URDF and controller config — not for the policy itself |
| `urdf_path` | *(required)* | Bare robot URDF |
| `robot` | `h2` | As above |

`policy.launch.py` / `bridge.launch.py` take `policy_onnx_path` and `robot`.

Everything is an explicit path — nothing assumes where your URDF or training runs live. Every
launch file derives the joint list the same way, from `deploy.yaml`'s `joint_ids_map` against
`robots/<name>.yaml`'s `joint_sdk_names`, so the effort array lines up with the
`effort_controller`'s `joints:` order by construction.

### What you should see

1. Gazebo opens **paused**, and the robot appears already in the trained crouch (not a
   straight-legged zero pose). It should be motionless.
2. A fraction of a second of simulation is stepped through so the controllers can activate —
   activation is serviced by the `controller_manager` update loop, so it needs steps.
3. `gz_reset_node` places the robot in `deploy.yaml`'s exact start state and the world
   un-pauses:
   ```
   Startup reset complete (31 joints reset to default pose) after 0.100s of simulation
   Reset received - re-seeding on post-reset data.
   Starting the trained policy from the reset default pose.
   ```
4. The policy takes over immediately and the robot stands. With no velocity command it holds
   position; `proj_grav_b` stays near `[0,0,-1]`.

## Driving the robot

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

Commands are clamped to the ranges the policy was trained on, read from `deploy.yaml`. For the
current H2 checkpoint:

| Field | Trained range |
| --- | --- |
| `linear.x` | -0.5 … 1.0 m/s |
| `linear.y` | -0.3 … 0.3 m/s |
| `angular.z` | -0.2 … 0.2 rad/s |

Anything outside is silently clamped. Stop publishing and the robot returns to standing.

Measured on the H2 checkpoint via `all.launch.py`, `linear.x = 0.5` held for 25 s: **0.408 m/s
with `unitree_isaac_policy`, 0.356 m/s with `unitree_policy_bridge`** — run-to-run variation is
larger than the gap between the two stacks. Yaw tracking is poor, roughly 0.03 rad/s achieved
for 0.2 commanded. That weakness looks like a property of this training run rather than of
either stack: the command reaches the observation vector correctly, which the `cmd=[...]` field
of the diagnostic log confirms.

## Resetting mid-run

`gz_reset_node` exposes its reset as a service, returning the robot to the trained start state
without restarting Gazebo:

```bash
ros2 service call /gz_reset_node/reset std_srvs/srv/Trigger
# -> success=True, message='31 joints reset to default pose'
```

This restores joint positions, joint velocities, base pose and base velocity, and publishes
`/policy_reset` so the policy drops its observation history and re-seeds on post-reset data.
Verified recovering from both standing and walking.

## Adding another robot

For a robot whose `deploy.yaml` uses observation and action terms the stack already supports,
add one file — `unitree_gz_description/robots/<name>.yaml`:

```yaml
imu_link: <link the IMU attaches to>
spawn_height: <base height of the standing pose, metres>
joint_sdk_names:      # the robot's SDK motor order, one name per motor
  - ...
```

Then launch with `robot:=<name>` and the new `urdf_path`/`policy_onnx_path`. No code changes.

A new **observation term type** is the one thing that does need code, and where it goes depends
on the stack:

- `unitree_isaac_policy` — nowhere, if upstream already registers it. It inherits everything in
  `unitree_rl_lab/deploy/include/isaaclab/envs/mdp/observations/observations.h`, `gait_phase`
  and `joint_pos` included. A genuinely new term is a `REGISTER_OBSERVATION` block, or an
  assignment into `observations_map()` the way `policy_node` overrides `velocity_commands`.
- `unitree_policy_bridge` — a branch in `observation_term_fn()` in
  `src/policy_bridge_node.cpp`. It implements only the six terms H2 uses and throws on anything
  else, `gait_phase` included.

**Before running a new robot, check gain/joint-order parity**: assert each joint's
`stiffness`/`damping` against the actuator groups in its training config, by name. The first
note below says why.

## Notes and gotchas

- **`deploy.yaml` is not uniformly ordered.** `stiffness` and `damping` are exported in the
  robot's SDK motor order, while `default_joint_pos`, the action scale/offset/clip and the
  observation scales are in Isaac's environment joint order. Reading them all positionally
  silently gives each joint another joint's gains — on H2 that put a knee at kp=40 instead of
  200, which cannot hold up a 75 kg robot. Both stacks re-index through `joint_ids_map`
  (`load_gains()` in `pd.hpp`, `load_dynamics()` in the bridge). Worth knowing before trusting
  any other array in that file.
- **Start the policy before the simulator** — see Launch files. Both stacks fail the same way
  if the world un-pauses with nothing commanding effort.
- **The two stacks are mutually exclusive.** Both publish to `/effort_controller/commands`.
- **`pd_node` does not clear its target on reset**, where `policy_bridge_node` does. During the
  re-seed window it keeps driving toward the pre-teleport target. Harmless at current sensor
  rates — the window is 1–2 ms and `gz_reset_node` keeps the world paused across the teleport —
  but it is mitigated by timing rather than by design.
- **The control loop runs on simulation time.** The launch bridges `/clock` and every node sets
  `use_sim_time`. Without the clock bridge the policy would step on wall time, at a rate
  unrelated to how fast physics is advancing.
- **The start pose comes from `initial_value`.** `generate_urdf.py` declares each joint's
  `default_joint_pos` as a nested `<param name="initial_value">` on its position state
  interface, which `gz_ros2_control` applies as a joint teleport at spawn. It must be a nested
  element — `ros2_control` silently ignores it as an attribute.
- **The world file is custom** (`unitree_gz_bringup/worlds/default.sdf`). Gazebo's stock
  `empty.sdf` doesn't load `gz-sim-imu-system`, so an IMU sensor never produces data on it.
- **There is no stand-up phase.** The reset establishes the trained start state directly, so the
  policy runs from the first step, as it does in Isaac Lab. On real hardware, where nothing can
  teleport the robot, reaching the default pose is the controller FSM's job upstream of these
  nodes; `wait_for_reset` (a parameter on both policy nodes, default `false`, set `true` by the
  launch files) is what makes the simulation wait for the teleport instead.
- **`IMU sensor 'imu_sensor' not found in hardware_info` is expected.** The IMU is a plain
  gz-sim sensor bridged by `ros_gz_bridge`, not a `gz_ros2_control` hardware-interface sensor.

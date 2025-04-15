from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, UsdFileCfg, RigidBodyPropertiesCfg,ArticulationRootPropertiesCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass
from .waypoint import WAYPOINT_CFG
from .leatherback import LEATHERBACK_CFG
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns

@configclass
class ChargingRobotEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 200.0
    action_space = 2
    observation_space = 8
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)
    robot_cfg: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    waypoint_cfg = WAYPOINT_CFG

    throttle_dof_name = [
        "Wheel__Knuckle__Front_Left",
        "Wheel__Knuckle__Front_Right",
        "Wheel__Upright__Rear_Right",
        "Wheel__Upright__Rear_Left"
    ]
    steering_dof_name = [
        "Knuckle__Upright__Front_Right",
        "Knuckle__Upright__Front_Left",
    ]

    env_spacing = 32.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=env_spacing, replicate_physics=True)

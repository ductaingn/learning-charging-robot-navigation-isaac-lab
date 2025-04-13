from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, UsdFileCfg, RigidBodyPropertiesCfg,ArticulationRootPropertiesCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass
from .waypoint import WAYPOINT_CFG
from isaaclab_assets.robots.leatherback import LEATHERBACK_CFG
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
    lidar_cfg: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/Rigid_Bodies/Chassis/Lidar",
        update_period=1 / 60,
        mesh_prim_paths=["/World/ground"],
        offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 0.5)),
        attach_yaw_only=True,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=10,
            vertical_fov_range=[-90, 90],
            horizontal_fov_range=[-181, 179],
            horizontal_res=1
        ),
        debug_vis=True
    )
    

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
# @configclass
# class ChargingRobotEnvCfg(DirectRLEnvCfg):
#     decimation = 4
#     episode_length_s = 20.0
#     action_space = 2
#     observation_space = 7
#     state_space = 0
#     sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)
#     robot_cfg: ArticulationCfg = ArticulationCfg(
#         prim_path='/World/envs/env_.*/Robot',
#         spawn=UsdFileCfg(
#             usd_path="/home/cei/Downloads/CEI_Isaacsim/Nova_Carter_ROS.usd",
#             rigid_props=RigidBodyPropertiesCfg(
#                 rigid_body_enabled=True,
#                 max_linear_velocity=1000.0,
#                 max_angular_velocity=1000.0,
#                 max_depenetration_velocity=100.0,
#                 enable_gyroscopic_forces=True,
#             ),
#             articulation_props=ArticulationRootPropertiesCfg(
#                 enabled_self_collisions=False,
#                 solver_position_iteration_count=4,
#                 solver_velocity_iteration_count=0,
#                 sleep_threshold=0.005,
#                 stabilization_threshold=0.001,
#             ),
#         ),

#         init_state=ArticulationCfg.InitialStateCfg(
#             pos=(0.0, 0.0, 0.05),
#             joint_pos={
#                 "joint_caster_base": 0.0,
#                 "joint_swing_left": 0.0,
#                 "joint_swing_right": 0.0,
#                 "joint_caster_left": 0.0,
#                 "joint_caster_right": 0.0,
#                 "joint_wheel_left": 0.0,
#                 "joint_wheel_right": 0.0
#             },
#         ),
#         actuators={
#             "throttle": ImplicitActuatorCfg(
#                 joint_names_expr=["joint_wheel.*"],
#                 effort_limit=40000.0,
#                 velocity_limit=100.0,
#                 stiffness=0.0,
#                 damping=100000.0,
#             )
#         },
#     )

#     lidar_cfg: RayCasterCfg = RayCasterCfg(
#         prim_path="/World/envs/env_.*/Robot/chassis_link",
#         update_period=1 / 60,
#         mesh_prim_paths=["/World/ground"],
#         offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 0.5)),
#         attach_yaw_only=True,
#         pattern_cfg=patterns.LidarPatternCfg(
#             channels=100, vertical_fov_range=[-90, 90], horizontal_fov_range=[-90, 90], horizontal_res=1.0
#         ),
#         debug_vis=True
#     )
#     waypoint_cfg = WAYPOINT_CFG

#     throttle_dof_name = [
#         "joint_wheel_left",
#         "joint_wheel_right"
#     ]

#     env_spacing = 32.0
#     scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=env_spacing, replicate_physics=True)
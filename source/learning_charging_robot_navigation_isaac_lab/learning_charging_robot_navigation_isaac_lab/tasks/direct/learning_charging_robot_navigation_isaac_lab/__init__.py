import gymnasium as gym

from . import agents


gym.register(
    id="Learning-Charging-Robot-Navigation-Direct-v0",
    entry_point=f"{__name__}.charging_robot_env:ChargingRobotEnv",
    disable_env_checker=False,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.charging_robot_env_cfg:ChargingRobotEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)
from .airsim_env import AirSimCarEnv, TestEnv
from gym.envs.registration import register


# register AirSim environment as a gym environment
register(
    id="airsim-env-v0", entry_point="scripts:AirSimCarEnv",
)

# register AirSim environment as a gym environment
register(
    id="test-env-v0", entry_point="scripts:TestEnv", # si riferisce sempre al file sopra citato
)

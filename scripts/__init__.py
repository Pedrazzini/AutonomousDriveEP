from .airsim_env import AirSimCarEnv, TestEnv
from gym.envs.registration import register


# Register AirSim environment as a gym environment
register(
    id="airsim-env-v0", entry_point="scripts:AirSimCarEnv", #si riferisce al nome della classe definita in airsim_env.py
)

# Register AirSim environment as a gym environment
register(
    id="test-env-v0", entry_point="scripts:TestEnv", # si riferisce sempre al file sopra citato
)

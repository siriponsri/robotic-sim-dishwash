"""Custom ManiSkill environments for the Unitree G1 robotics project.

Environments
------------
- UnitreeG1DishWipeEnv              : Upper-body (25 DOF) dish-wiping (reference)
- UnitreeG1PlaceAppleInBowlFullBodyEnv : Full-body (37 DOF) apple-in-bowl (main)
- UnitreeG1DishWipeFullBodyEnv      : Full-body (37 DOF) dish-wiping (bonus)
"""

from src.envs.dishwipe_env import UnitreeG1DishWipeEnv  # noqa: F401
from src.envs.apple_fullbody_env import UnitreeG1PlaceAppleInBowlFullBodyEnv  # noqa: F401
from src.envs.dishwipe_fullbody_env import UnitreeG1DishWipeFullBodyEnv  # noqa: F401

__all__ = [
    "UnitreeG1DishWipeEnv",
    "UnitreeG1PlaceAppleInBowlFullBodyEnv",
    "UnitreeG1DishWipeFullBodyEnv",
]

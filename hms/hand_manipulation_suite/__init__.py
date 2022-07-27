from gym.envs.registration import register

register(
    id='door-v0',
    entry_point='hand_manipulation_suite:DoorEnvV0',
    max_episode_steps=200,
)
from hand_manipulation_suite.door_v0 import DoorEnvV0

register(
    id='generalized-door-v0',
    entry_point='hand_manipulation_suite:GeneralizedDoorEnvV0',
    max_episode_steps=200,
)
from hand_manipulation_suite.generalized_door import GeneralizedDoorEnvV0

register(
    id='relocate-v0',
    entry_point='hand_manipulation_suite:RelocateEnvV0',
    max_episode_steps=200,
)
from hand_manipulation_suite.relocate_v0 import RelocateEnvV0

register(
    id='generalized-relocate-v0',
    entry_point='hand_manipulation_suite:GeneralizedRelocateEnvV0',
    max_episode_steps=200,
)
from hand_manipulation_suite.generalized_relocate import GeneralizedRelocateEnvV0

register(
    id='hammer-v0',
    entry_point='hand_manipulation_suite:HammerEnvV0',
    max_episode_steps=200,
)
from hand_manipulation_suite.hammer_v0 import HammerEnvV0

register(
    id='generalized-hammer-v0',
    entry_point='hand_manipulation_suite:GeneralizedHammerEnvV0',
    max_episode_steps=200,
)
from hand_manipulation_suite.generalized_hammer import GeneralizedHammerEnvV0


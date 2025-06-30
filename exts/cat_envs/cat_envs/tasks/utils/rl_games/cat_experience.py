from rl_games.common.experience import VectorizedReplayBuffer, ExperienceBuffer
import torch
import gym
import numpy as np


class CaTVectorizedReplayBuffer(VectorizedReplayBuffer):
    """
    This class redefines some variables of VectorizedReplayBuffer as float instead of
    integers since CaT relies on float terminations instead of simply 0 or 1.
    """

    def __init__(self, obs_shape, action_shape, capacity, device):
        super().__init__()

        # Redefining dones as float32 instead of uint8
        self.dones = torch.empty((capacity, 1), dtype=torch.float32, device=self.device)


class CaTExperienceBuffer(ExperienceBuffer):
    """
    This class redefines some variables of ExperienceBuffer as float instead of
    integers since CaT relies on float terminations instead of simply 0 or 1.
    """

    def _init_from_env_info(self, env_info):
        super()._init_from_env_info(env_info)

        # Redefining dones as float32 instead of uint8
        self.tensor_dict["dones"] = self._create_tensor_from_space(
            gym.spaces.Box(low=0, high=1, shape=(), dtype=np.float32),
            self.obs_base_shape,
        )

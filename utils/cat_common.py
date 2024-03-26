# from rl_games.common.a2c_common import A2CBase
# class CaTA2CBase(A2CBase):

from rl_games.algos_torch import a2c_continuous
from rl_games.common.a2c_common import swap_and_flatten01
import torch
import time
from utils.cat_experience import CaTExperienceBuffer


class CaTA2CAgent(a2c_continuous.A2CAgent):
    """
    This class redefines some variables of A2CBase as float instead of integers
    since CaT relies on float terminations instead of simply 0 or 1.
    """

    def init_tensors(self):
        super().init_tensors()

        batch_size = self.num_agents * self.num_actors
        algo_info = {
            "num_actors": self.num_actors,
            "horizon_length": self.horizon_length,
            "has_central_value": self.has_central_value,
            "use_action_masks": self.use_action_masks,
        }

        # Experience buffer that uses float32 dones
        self.experience_buffer = CaTExperienceBuffer(
            self.env_info, algo_info, self.ppo_device
        )

        # Redefining dones as float32 instead of uint8
        self.dones = torch.ones(
            (batch_size,), dtype=torch.float32, device=self.ppo_device
        )

    def play_steps(self):
        update_list = self.update_list

        step_time = 0.0

        for n in range(self.horizon_length):
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            self.experience_buffer.update_data("obses", n, self.obs["obs"])
            self.experience_buffer.update_data("dones", n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            if self.has_central_value:
                self.experience_buffer.update_data("states", n, self.obs["states"])

            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict["actions"])
            step_time_end = time.time()

            step_time += step_time_end - step_time_start

            shaped_rewards = self.rewards_shaper(rewards)

            if self.value_bootstrap and "time_outs" in infos:
                shaped_rewards += (
                    self.gamma
                    * res_dict["values"]
                    * self.cast_obs(infos["time_outs"]).unsqueeze(1).float()
                )

            self.experience_buffer.update_data("rewards", n, shaped_rewards)

            self.dones = self.dones.float()  # <-- Modified line CaT
            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1
            all_done_indices = self.dones.ge(1.0).nonzero(
                as_tuple=False
            )  # <-- Modified line CaT
            env_done_indices = all_done_indices[:: self.num_agents]

            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_shaped_rewards.update(
                self.current_shaped_rewards[env_done_indices]
            )
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones  # <-- Modified line CaT

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_shaped_rewards = (
                self.current_shaped_rewards * not_dones.unsqueeze(1)
            )
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs)

        mb_fdones = self.experience_buffer.tensor_dict["dones"]
        mb_values = self.experience_buffer.tensor_dict["values"]
        mb_rewards = self.experience_buffer.tensor_dict["rewards"]
        mb_advs = self.discount_values(
            self.dones, last_values, mb_fdones, mb_values, mb_rewards
        )
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(
            swap_and_flatten01, self.tensor_list
        )
        batch_dict["returns"] = swap_and_flatten01(mb_returns)
        batch_dict["played_frames"] = self.batch_size
        batch_dict["step_time"] = step_time

        return batch_dict

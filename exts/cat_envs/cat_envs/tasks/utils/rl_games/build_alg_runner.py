from rl_games.algos_torch import players
from cat_envs.tasks.utils.rl_games.cat_common import CaTA2CAgent

from rl_games.torch_runner import Runner
from rl_games.algos_torch import model_builder


def build_alg_runner(algo_observer=None):
    if algo_observer == None:
        runner = Runner()
    else:
        runner = Runner(algo_observer)

    runner.algo_factory.register_builder(
        "cat_a2c_continuous", lambda **kwargs: CaTA2CAgent(**kwargs)
    )
    runner.player_factory.register_builder(
        "cat_a2c_continuous", lambda **kwargs: players.PpoPlayerContinuous(**kwargs)
    )

    return runner
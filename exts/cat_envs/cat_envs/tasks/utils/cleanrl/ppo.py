import os
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


class RunningMeanStd(nn.Module):
    def __init__(self, shape=(), epsilon=1e-08):
        super(RunningMeanStd, self).__init__()
        self.register_buffer("running_mean", torch.zeros(shape))
        self.register_buffer("running_var", torch.ones(shape))
        self.register_buffer("count", torch.ones(()))

        self.epsilon = epsilon

    def forward(self, obs, update=True):
        if update:
            self.update(obs)

        return (obs - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, correction=0, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.running_mean, self.running_var, self.count = (
            update_mean_var_count_from_moments(
                self.running_mean,
                self.running_var,
                self.count,
                batch_mean,
                batch_var,
                batch_count,
            )
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        SINGLE_OBSERVATION_SPACE = envs.unwrapped.single_observation_space[
            "policy"
        ].shape
        SINGLE_ACTION_SPACE = envs.unwrapped.single_action_space.shape
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(SINGLE_OBSERVATION_SPACE).prod(), 512)),
            nn.ELU(),
            layer_init(nn.Linear(512, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(SINGLE_OBSERVATION_SPACE).prod(), 512)),
            nn.ELU(),
            layer_init(nn.Linear(512, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, np.prod(SINGLE_ACTION_SPACE)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(SINGLE_ACTION_SPACE)))

        self.obs_rms = RunningMeanStd(shape=SINGLE_OBSERVATION_SPACE)
        self.value_rms = RunningMeanStd(shape=())

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, deterministic=False):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            if not deterministic:
                action = probs.sample()
            else:
                action = action_mean
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )
    
    def forward(self, x, deterministic=True):
        action, _, _, _ = self.get_action_and_value(self.obs_rms(x, update=False), deterministic=deterministic)
        return action


def PPO(envs, ppo_cfg, run_path):
    if ppo_cfg.logger == "wandb":
        from rsl_rl.utils.wandb_utils import WandbSummaryWriter

        writer = WandbSummaryWriter(
            log_dir=run_path, flush_secs=10, cfg=ppo_cfg.to_dict()
        )
    elif ppo_cfg.logger == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

        writer = TensorboardSummaryWriter(log_dir=run_path)
    else:
        raise AssertionError("logger type not found")

    if not os.path.exists(run_path):
        os.makedirs(run_path)

    LEARNING_RATE = ppo_cfg.learning_rate
    NUM_STEPS = ppo_cfg.num_steps
    NUM_ITERATIONS = ppo_cfg.num_iterations
    GAMMA = ppo_cfg.gamma
    GAE_LAMBDA = ppo_cfg.gae_lambda
    UPDATES_EPOCHS = ppo_cfg.updates_epochs
    MINIBATCH_SIZE = ppo_cfg.minibatch_size
    CLIP_COEF = ppo_cfg.clip_coef
    ENT_COEF = ppo_cfg.ent_coef
    VF_COEF = ppo_cfg.vf_coef
    MAX_GRAD_NORM = ppo_cfg.max_grad_norm
    NORM_ADV = ppo_cfg.norm_adv
    CLIP_VLOSS = ppo_cfg.clip_vloss
    ANNEAL_LR = ppo_cfg.anneal_lr

    NUM_ENVS = envs.unwrapped.num_envs

    SINGLE_OBSERVATION_SPACE = envs.unwrapped.single_observation_space["policy"].shape
    SINGLE_ACTION_SPACE = envs.unwrapped.single_action_space.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = int(NUM_ENVS * NUM_STEPS)

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    obs = torch.zeros(
        (NUM_STEPS, NUM_ENVS) + SINGLE_OBSERVATION_SPACE, dtype=torch.float
    ).to(device)
    actions = torch.zeros(
        (NUM_STEPS, NUM_ENVS) + SINGLE_ACTION_SPACE, dtype=torch.float
    ).to(device)
    logprobs = torch.zeros((NUM_STEPS, NUM_ENVS), dtype=torch.float).to(device)
    rewards = torch.zeros((NUM_STEPS, NUM_ENVS), dtype=torch.float).to(device)
    dones = torch.zeros((NUM_STEPS, NUM_ENVS), dtype=torch.float).to(device)
    true_dones = torch.zeros((NUM_STEPS, NUM_ENVS), dtype=torch.float).to(device)
    values = torch.zeros((NUM_STEPS, NUM_ENVS), dtype=torch.float).to(device)
    advantages = torch.zeros_like(rewards, dtype=torch.float).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()[0]["policy"]
    next_obs = agent.obs_rms(next_obs)
    next_done = torch.zeros(NUM_ENVS, dtype=torch.float).to(device)
    next_true_done = torch.zeros(NUM_ENVS, dtype=torch.float).to(device)

    print(f"Starting training for {NUM_ITERATIONS} steps")

    for iteration in range(1, NUM_ITERATIONS + 1):
        ep_infos = []

        if ANNEAL_LR:
            frac = 1.0 - (iteration - 1.0) / NUM_ITERATIONS
            lrnow = frac * LEARNING_RATE
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, NUM_STEPS):
            global_step += NUM_ENVS
            obs[step] = next_obs
            dones[step] = next_done
            true_dones[step] = next_true_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards[step], next_done, timeouts, info = envs.step(action)
            next_done = next_done.to(torch.float)
            next_obs = next_obs["policy"]

            if "episode" in info:
                ep_infos.append(info["episode"])
            elif "log" in info:
                ep_infos.append(info["log"])

            info["true_dones"] = timeouts
            next_obs = agent.obs_rms(next_obs)
            next_true_done = info["true_dones"].float()
            if "time_outs" in info:
                if info["time_outs"].any():
                    print("time outs", info["time_outs"].sum())
                    exit(0)

        # Adapted from rslrl
        for key in ep_infos[0]:
            infotensor = torch.tensor([], device=device)
            for ep_info in ep_infos:
                # handle scalar and zero dimensional tensor infos
                if key not in ep_info:
                    continue
                if not isinstance(ep_info[key], torch.Tensor):
                    ep_info[key] = torch.Tensor([ep_info[key]])
                if len(ep_info[key].shape) == 0:
                    ep_info[key] = ep_info[key].unsqueeze(0)
                infotensor = torch.cat((infotensor, ep_info[key].to(device)))
            value = torch.mean(infotensor)
            if "/" in key:
                writer.add_scalar(key, value, iteration)
            else:
                writer.add_scalar("Episode/" + key, value, iteration)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
                    true_nextnonterminal = 1 - next_true_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    true_nextnonterminal = 1 - true_dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t]
                    + GAMMA * nextvalues * nextnonterminal * true_nextnonterminal
                    - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + GAMMA
                    * GAE_LAMBDA
                    * nextnonterminal
                    * true_nextnonterminal
                    * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + SINGLE_OBSERVATION_SPACE)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + SINGLE_ACTION_SPACE)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_values = agent.value_rms(b_values)
        b_returns = agent.value_rms(b_returns)

        sum_pg_loss = sum_entropy_loss = sum_v_loss = sum_surrogate_loss = 0.0

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(UPDATES_EPOCHS):
            b_inds = torch.randperm(BATCH_SIZE, device=device)
            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > CLIP_COEF).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if NORM_ADV:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - CLIP_COEF, 1 + CLIP_COEF
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                newvalue = agent.value_rms(newvalue, update=False)
                if CLIP_VLOSS:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -CLIP_COEF,
                        CLIP_COEF,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VF_COEF

                sum_pg_loss += pg_loss
                sum_entropy_loss += entropy_loss
                sum_v_loss += v_loss
                sum_surrogate_loss += loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        num_updates = UPDATES_EPOCHS * BATCH_SIZE / MINIBATCH_SIZE
        writer.add_scalar("Loss/mean_pg_loss", sum_pg_loss / num_updates, iteration)
        writer.add_scalar(
            "Loss/mean_entropy_loss", sum_entropy_loss / num_updates, iteration
        )
        writer.add_scalar("Loss/mean_v_loss", sum_v_loss / num_updates, iteration)
        writer.add_scalar(
            "Loss/mean_surrogate_loss", sum_surrogate_loss / num_updates, iteration
        )
        writer.add_scalar(
            "Loss/learning_rate", optimizer.param_groups[0]["lr"], iteration
        )

        if (iteration + 1) % ppo_cfg.save_interval == 0:
            model_path = f"{run_path}/model_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print("Saved model")

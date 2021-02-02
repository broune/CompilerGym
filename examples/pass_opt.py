# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Simple PT compiler gym actor-critic RL example.

Usage: python actor_critic.py

Use --help to list the configurable options.

The objective is to minimize the size of a benchmark (program) using
LLVM compiler passes. At each step there is a choice of which pass to
pick next and an episode consists of a sequence of such choices,
yielding the number of saved instructions as the overall reward.

For simplification of the learning task, only a (configurable) subset
of LLVM passes are considered and every episode has the same
(configurable) length.

Based on the PT actor-critic example:
https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
"""
import hashlib as hashlib
import random
import statistics
from collections import namedtuple
from math import log
from statistics import mean
from typing import List

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from absl import app, flags
from torch.distributions import Categorical

import compiler_gym  # Register environments.
from compiler_gym.util.flags.benchmark_from_flags import benchmark_from_flags
from compiler_gym.util.flags.env_from_flags import env_from_flags

flags.DEFINE_list(
    "flags",
    [
        "-early-cse-memssa",
        "-flattencfg",
        "-gvn",
        "-gvn-hoist",
        "-inferattrs",
        "-instcombine",
        "-instsimplify",
        "-jump-threading",
        "-load-store-vectorizer",
        "-loop-extract",
        "-loop-reduce",
        "-loop-simplifycfg",
        "-loop-versioning",
        "-mem2reg",
        "-newgvn",
        "-reg2mem",
        "-simplifycfg",
        "-sroa",
        "-structurizecfg",
    ],
    "List of optimizatins to explore.",
)

flags.DEFINE_integer("episode_len", 5, "Number of transitions per episode.")
flags.DEFINE_integer("hidden_size", 64, "Latent vector size.")
flags.DEFINE_integer("episodes", 1000, "Number of episodes to run and train on.")
flags.DEFINE_integer("log_interval", 100, "Episodes per log output.")
flags.DEFINE_integer("seed", 101, "Seed for all pseudo-random choices.")
flags.DEFINE_integer("iterations", 1, "Times to redo entire training.")
flags.DEFINE_float("learning_rate", 0.008, "The learning rate for training.")
flags.DEFINE_float("exploration", 0.0, "Rate to explore random transitions.")
flags.DEFINE_float("mean_smoothing", 0.95, "Smoothing factor for mean normalization.")
flags.DEFINE_float("std_smoothing", 0.4, "Smoothing factor for std dev normalization.")
flags.DEFINE_bool("verbose_eval", False, "Print out all eval results.")
flags.DEFINE_bool("use_ap", False, "Use autophase features.")
flags.DEFINE_bool("unique_actions", True, "Ban doing an action more than once per eps.")
flags.DEFINE_bool("use_prob_max_exp", True, "Do exploration using max on probs.")
flags.DEFINE_bool("use_training_set", True, "Train across benchmarks instead of one.")

FLAGS = flags.FLAGS

# hash() returns a different hash between runs.
def deterministic_hash(x):
    return int(hashlib.sha1(x.encode()).hexdigest(), 16)


data_set = [f"benchmark://npb-v0/{n}" for n in range(1, 122)]
training_set = [x for x in data_set if deterministic_hash(x) % 100 < 80]
validation_set = [x for x in data_set if x not in training_set]


eps = np.finfo(np.float32).eps.item()


class MovingExponentialAverage:
    """Simple class to calculate exponential moving averages."""

    def __init__(self, smoothing_factor):
        self.smoothing_factor = smoothing_factor
        self.value = None

    def next(self, entry):
        assert entry is not None
        if self.value is None:
            self.value = entry
        else:
            self.value = (
                entry * (1 - self.smoothing_factor) + self.value * self.smoothing_factor
            )
        return self.value


# Like a CompilerGym env but takes a subset of actions and limits the
# episode length to the flag value. That's it, should probably be
# gotten rid of and this logic moving elsewhere.
class CustomEnv:
    def __init__(self):
        self._env = env_from_flags(benchmark_from_flags())

        try:
            self._env.require_dataset("cBench-v0")
            self._env.reset()
            self.benchmark = self._env.benchmark

            # Project onto the subset of transformations that have
            # been specified to be used.
            self._actions = [self._env.action_space.flags.index(f) for f in FLAGS.flags]
            self.action_space = [self._env.action_space.flags[a] for a in self._actions]

        except:
            # The program will not terminate until the environment is
            # closed, not even if there is an exception.
            self._env.close()
            raise

    def step(self, action):
        assert 0 <= action and action < len(self._actions)

        # Update environment
        state, reward, done, info = self._env.step(self._actions[action])
        self._steps_taken += 1
        done = done or self._steps_taken == FLAGS.episode_len

        return state, reward, done, info

    @property
    def observation(self):
        return self._env.observation

    def reset(self, benchmark=None):
        if benchmark is None:
            benchmark = self.benchmark
        else:
            self.benchmark = benchmark
            self._env.benchmark = benchmark
        self._steps_taken = 0
        return self._env.reset()

    def close(self):
        self._env.close()


# A trajectory, saving which actions were taken and some properties
# computed along the way.
class Trajectory:
    SavedAction = namedtuple("Action", ["action", "probs", "value"])

    def __init__(self):
        self.actions: List[self.SavedAction] = []
        self.rewards: List[int] = []

    def reward_sum(self):
        return sum(a.reward for a in self.actions)

    def taken_probs(self):
        return [a.probs[a.action] for a in self.actions]

    def values(self):
        return [a.value for a in self.actions]

    def rewards(self):
        return self.rewards

    def total_reward(self):
        return sum(self.rewards)

    def returns(self, discount_factor=1.0):
        R = 0.0  # cannot call this "return" as that is a keyword
        returns = [0.0] * self.len()

        # Calculate return going from end of trajectory to beginning.
        for i in reversed(range(self.len())):
            R = R * discount_factor + self.rewards[i]
            returns[i] = R
        return returns

    def len(self):
        return len(self.actions)

    def histogram(self, action_count):
        h = [0] * action_count
        for a in self.actions:
            h[a.action] += 1
        return h


class Policy(nn.Module):
    """A very simple actor critic policy model."""

    def preprocess(self, trajectory, env):
        action_count = len(FLAGS.flags)
        episode_len = FLAGS.episode_len

        assert trajectory.len() < episode_len

        ap = list(env.observation["Autophase"])
        assert len(ap) == 56

        return (
            torch.FloatTensor(trajectory.histogram(action_count)),
            torch.FloatTensor(ap),
        )

    def __init__(self):
        super().__init__()

        input_size = len(FLAGS.flags) + (56 if FLAGS.use_ap else 0)
        self.affine1 = nn.Linear(input_size, FLAGS.hidden_size)
        self.affine2 = nn.Linear(FLAGS.hidden_size, FLAGS.hidden_size)
        self.affine3 = nn.Linear(FLAGS.hidden_size, FLAGS.hidden_size)
        self.affine4 = nn.Linear(FLAGS.hidden_size, FLAGS.hidden_size)

        # Actor's layer
        self.action_head = nn.Linear(FLAGS.hidden_size, len(FLAGS.flags))

        # Critic's layer
        self.value_head = nn.Linear(FLAGS.hidden_size, 1)

        # Keep exponential moving average of mean and standard
        # deviation for use in normalization of the return.
        self.return_moving_mean = MovingExponentialAverage(FLAGS.mean_smoothing)
        self.return_moving_std = MovingExponentialAverage(FLAGS.std_smoothing)

    def forward(self, args):
        histogram, ap = args

        if FLAGS.use_ap:
            # Feature 51 is the number of instructions and this
            # normalization is the one suggested in the AP paper.
            ap = ap / ap[51]

        x = histogram
        if FLAGS.use_ap:
            # TODO: Use a hyperparameter that is initially
            # down-weighting ap over the histogram as the histogram is
            # initially more important.
            x = torch.cat((x, ap))

        """Forward of both actor and critic"""
        # Initial layer maps the sequence of one-hot vectors into a
        # vector of the hidden size. Next layers stay with the same
        # size and use residual connections.
        x = F.relu(self.affine1(x))
        x = x.add(F.relu(self.affine2(x)))
        x = x.add(F.relu(self.affine3(x)))
        x = x.add(F.relu(self.affine4(x)))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        if FLAGS.unique_actions:
            # If this is enabled, the histogram will be 0-1, so this
            # will zero the probability of the actions that were
            # already taken.
            action_prob = action_prob * (1 - histogram)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


def select_action(model, trajectory, env, exploration_rate=0.0, greedy=False):
    """Selects an action and registers it with the action buffer."""
    input = model.preprocess(trajectory, env)
    probs, state_value = model(input)

    if greedy is True:
        action = torch.argmax(probs)
    else:
        if FLAGS.use_prob_max_exp and exploration_rate > eps:
            # set each action to have a minimum probability with
            # max. This has pros and cons. It's both good and bad that
            # a max that takes a constant parameter will drive a
            # gradient to the constant (which is ignored) instead of
            # the calculated probability. The probability of
            # exploration will also be actually a bit lower after
            # normalization to sum to 1.
            rate = torch.Tensor([exploration_rate / len(probs)])
            probs = torch.max(rate, probs - rate)

        # Create a probability distribution where the probability of
        # action i is probs[i].
        m = Categorical(probs)

        # Sample an action using the distribution, or pick an action
        # uniformly at random if in an exploration mode.
        if (not FLAGS.use_prob_max_exp) and random.random() < exploration_rate:
            action = torch.tensor(random.randrange(0, len(probs)))
        else:
            action = m.sample()

        # TODO: Don't know why this makes a difference, but it
        # does. Maybe just numerics, but seems a bigger effect that
        # than. Maybe normalization?
        probs = m.probs

    trajectory.actions.append(
        Trajectory.SavedAction(action=action.item(), probs=probs, value=state_value)
    )

    return action.item()


def finish_episode(trajectory, model, optimizer) -> float:
    """The training code. Calculates actor and critic loss and performs backprop."""
    # TODO: try an L1 or L2 loss on all the weights
    #
    # TODO: Try clipping rewards since a few benchmarks have huge
    # deviations, like +12 or -17 when most are +-1.05.

    R = 0
    policy_losses = []  # list to save actor (policy) loss
    value_losses = []  # list to save critic (value) loss

    returns = trajectory.returns(0.95)

    # Update the moving averages for mean and standard deviation and
    # use that to normalize the input.
    returns = torch.tensor(returns)
    model.return_moving_mean.next(returns.mean())
    model.return_moving_std.next(returns.std())
    returns = (returns - model.return_moving_mean.value) / (
        model.return_moving_std.value + eps
    )

    for prob, value, R in zip(trajectory.taken_probs(), trajectory.values(), returns):
        advantage = R - value.item()
        policy_losses.append(-torch.log(prob) * advantage)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # Reset gradients.
    optimizer.zero_grad()

    # Sum up all the values of policy_losses and value_losses.
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss_value = loss.item()

    # Perform backprop.
    loss.backward()
    optimizer.step()

    return loss_value


def DoEpisode(env, benchmark, model, greedy=False):
    env.reset(benchmark=benchmark)
    ep_reward = 0
    trajectory = Trajectory()

    # The environment keeps track of when the episode is done, so
    # we can loop infinitely here.
    while True:
        action = select_action(
            model, trajectory, env, exploration_rate=FLAGS.exploration, greedy=greedy
        )
        _, reward, done, _ = env.step(action)
        trajectory.rewards.append(reward)
        if done:
            break

    return trajectory


# Return average reward on the test set (well, validation set, since
# this is used for manual tuning).
def EvalActorCritic(env, model, benchmarks):
    total_rewards = []
    for b in benchmarks:
        trajectory = DoEpisode(env, b, model, greedy=True)
        total_rewards.append(trajectory.total_reward())

        if FLAGS.verbose_eval:
            print(
                trajectory.total_reward(),
                env.benchmark,
                [
                    f"{p.item():.2f}, {env.action_space[a.action]}"
                    for a, p in zip(trajectory.actions, trajectory.taken_probs())
                ],
            )

    return mean(total_rewards)


def TrainActorCritic(env, benchmark):
    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    # These statistics are just for logging.
    max_ep_reward = -float("inf")
    avg_reward = MovingExponentialAverage(0.95)
    avg_loss = MovingExponentialAverage(0.95)

    for episode in range(1, FLAGS.episodes + 1):
        if FLAGS.use_training_set:
            # Pick a random benchmark (using a very primitive "random" number).
            benchmark = training_set[episode * 101 % len(training_set)]

        trajectory = DoEpisode(env, benchmark, model)

        # Perform back propagation.
        loss = finish_episode(trajectory, model, optimizer)

        # Update statistics.
        ep_reward = trajectory.total_reward()
        max_ep_reward = max(max_ep_reward, ep_reward)
        avg_reward.next(ep_reward)
        avg_loss.next(loss)

        # Log statistics.
        if (
            episode == 1
            or episode % FLAGS.log_interval == 0
            or episode == FLAGS.episodes
        ):
            print(
                f"Episode {episode}\t"
                f"Last reward: {ep_reward:.2f}\t"
                f"Avg reward: {avg_reward.value:.2f}\t"
                f"Best reward: {max_ep_reward:.2f}\t"
                f"Last loss: {loss:.6f}\t"
                f"Avg loss: {avg_loss.value:.6f}\t",
                f"Eval: {EvalActorCritic(env, model, validation_set):.2f}",
                flush=True,
            )

    print(f"\nFinal performance (avg reward): {avg_reward.value:.2f}")
    print(f"Final avg reward versus own best: {avg_reward.value - max_ep_reward:.2f}")

    # One could also return the best found solution here, though that
    # is more random and noisy, while the average reward indicates how
    # well the model is working on a consistent basis.
    return avg_reward.value


def main(argv):
    """Main entry point."""
    torch.manual_seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    env = CustomEnv()
    try:
        print(f"Seed: {FLAGS.seed}")
        print(f"Episode length: {FLAGS.episode_len}")
        print(f"Exploration: {FLAGS.exploration:.2%}")
        print(f"Learning rate: {FLAGS.learning_rate}")
        print(f"Reward: {FLAGS.reward}")
        print(f"Benchmark: {FLAGS.benchmark}")
        print(f"Action space: {env.action_space}")

        if FLAGS.iterations == 1:
            TrainActorCritic(env, FLAGS.benchmark)
            return

        # Performance varies greatly with random initialization and
        # other random choices, so run the process multiple times to
        # determine the distribution of outcomes.
        performances = []
        for i in range(1, FLAGS.iterations + 1):
            print(f"\n*** Iteration {i} of {FLAGS.iterations}")
            performances.append(TrainActorCritic(env))

        print("\n*** Summary")
        print(f"Final performances: {performances}\n")
        print(f"  Best performance: {max(performances):.2f}")
        print(f"Median performance: {statistics.median(performances):.2f}")
        print(f"   Avg performance: {statistics.mean(performances):.2f}")
        print(f" Worst performance: {min(performances):.2f}")

    finally:
        # The program will not terminate until the environment is
        # closed.
        env.close()


if __name__ == "__main__":
    app.run(main)

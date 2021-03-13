import abc
import time
from copy import deepcopy
from collections import defaultdict


class BaseRLAgent(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_action(self, state):
        pass

    @abc.abstractmethod
    def update(self, trajectory, num_steps):
        pass

    def online_update(self, trajectory, num_steps):
        pass

    def additional_metrics(self, trajectory, num_steps, logged_episode_values):
        return None


class TrainingEngine:
    def __init__(self, env, agent, discount_rate, num_runs, num_episodes_per_run, save_frequency=5, max_steps=None):
        self.env = env
        self.agent = agent
        self.discount_rate = discount_rate
        self.num_runs = num_runs
        self.num_episodes_per_run = num_episodes_per_run
        self.save_frequency = save_frequency
        self.max_steps = max_steps

    def run(self):
        start_time = time.perf_counter()
        logged_run_values = defaultdict(list)

        for i_run in range(self.num_runs):
            logged_episode_values = defaultdict(list)

            # training loop
            env = deepcopy(self.env)
            agent = deepcopy(self.agent)
            state = env.reset()
            for i_ep in range(self.num_episodes_per_run):
                done = False
                trajectory = []
                total_reward = 0
                total_discounted_reward = 0
                steps = 0

                while not done:
                    prev_state = state

                    action = agent.get_action(state)
                    state, reward, done = env.step(action)

                    total_reward += reward
                    total_discounted_reward += reward * self.discount_rate**steps

                    trajectory.append((prev_state, action, reward))

                    agent.online_update(trajectory=trajectory, num_steps=steps)

                    steps += 1
                    if self.max_steps and steps >= self.max_steps:
                        break

                agent.update(trajectory=trajectory, num_steps=steps)

                # reset
                state = env.reset()
                print("ep", i_ep, "time", (time.perf_counter() - start_time) / 60, flush=True)

                if i_ep % self.save_frequency == 0:
                    logged_episode_values['returns'].append(total_reward)
                    logged_episode_values['discounted_returns'].append(total_discounted_reward)

                    additional_metrics = agent.additional_metrics(trajectory=trajectory,
                                                                  num_steps=steps,
                                                                  logged_episode_values=logged_episode_values)
                    if additional_metrics and isinstance(additional_metrics, dict):
                        for key, value in additional_metrics.items():
                            logged_episode_values[key].append(value)

                    # TODO understand the correct interface
                    # run evaluation phase 
                    # eval_logged = eval_agent(agent, env, n_test=100, max_steps=100)
                    # eval_state_visitation_entropy = eval_logged["state_visitation_entropy"]
                    # logged_run_values["state_visitation_entropy_eval"].append(eval_state_visitation_entropy)

            for key, value in logged_episode_values.items():
                logged_run_values[key].append(value)

        return logged_run_values

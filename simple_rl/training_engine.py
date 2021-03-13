import abc
import time
from inspect import signature
from collections import defaultdict


class BaseRLAgent(abc.ABC):
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
    # Defines the name of the environment argument in the agent class
    # if the agent class is to be initialized with the env
    AGENT_ENV_ARG = 'env'

    def __init__(self, env_class, env_parameters,
                 agent_class, agent_parameters,
                 discount_rate, num_runs,
                 num_iterations_per_run, iteration_counter='episodes',
                 pass_env_to_agent=False,
                 save_frequency=5, save_iterator='episodes',
                 max_steps=None):
        self.env_class = env_class
        self.env_parameters = env_parameters
        self.agent_class = agent_class
        self.agent_parameters = agent_parameters
        self.discount_rate = discount_rate
        self.num_runs = num_runs
        self.num_iterations_per_run = num_iterations_per_run
        self.iteration_counter = iteration_counter  # episodes or steps
        self.pass_env_to_agent = pass_env_to_agent
        self.save_frequency = save_frequency
        self.save_iterator = save_iterator  # episodes or steps
        self.max_steps = max_steps

        self.run_metrics = defaultdict(list)
        self.iterative_metrics = defaultdict(list)

        self._sanity_check()

    def run(self):
        start_time = time.perf_counter()
        self.run_metrics = defaultdict(list)

        for i_run in range(self.num_runs):
            self.iterative_metrics = defaultdict(list)  # reset for run
            env, agent = self._init_env_agent()

            total_num_steps = [0]
            i_ep = [0]
            counter = i_ep if self.iteration_counter == 'episodes' else total_num_steps
            break_value = self.num_iterations_per_run

            while counter[0] < break_value:
                state = env.reset()
                done = False
                trajectory = []
                total_reward = 0
                total_discounted_reward = 0
                num_steps = 0

                while not done:
                    prev_state = state

                    action = agent.get_action(state)
                    state, reward, done = env.step(action)

                    total_reward += reward
                    total_discounted_reward += reward * self.discount_rate**num_steps

                    trajectory.append((prev_state, action, reward))

                    agent.online_update(trajectory=trajectory, num_steps=num_steps)

                    if self.save_iterator == 'steps' and total_num_steps[0] % self.save_frequency == 0:
                        self._save_iterative_metrics(num_steps=num_steps, agent=agent, trajectory=trajectory,
                                                     total_reward=total_reward, total_discounted_reward=total_discounted_reward)

                    num_steps += 1
                    total_num_steps[0] += 1
                    if (self.max_steps and num_steps >= self.max_steps) or (counter[0] >= break_value):
                        break

                agent.update(trajectory=trajectory, num_steps=num_steps)
                print("ep", i_ep[0], "time", (time.perf_counter() - start_time) / 60, flush=True)

                if self.save_iterator == 'episodes' and i_ep[0] % self.save_frequency == 0:
                    self._save_iterative_metrics(num_steps=num_steps, agent=agent, trajectory=trajectory,
                                                 total_reward=total_reward, total_discounted_reward=total_discounted_reward)

                # TODO understand the correct interface
                # run evaluation phase 
                # eval_logged = eval_agent(agent, env, n_test=100, max_steps=100)
                # eval_state_visitation_entropy = eval_logged["state_visitation_entropy"]
                # logged_run_values["state_visitation_entropy_eval"].append(eval_state_visitation_entropy)
                i_ep[0] += 1

            for key, value in self.iterative_metrics.items():
                self.run_metrics[key].append(value)

        return self.run_metrics

    def _init_env_agent(self):
        env = self.env_class(**self.env_parameters)

        if self.pass_env_to_agent:
            self.agent_parameters.update({self.AGENT_ENV_ARG: env})

        agent = self.agent_class(**self.agent_parameters)

        return env, agent

    def _save_iterative_metrics(self, num_steps, agent, trajectory, total_reward, total_discounted_reward):
        self.iterative_metrics['returns'].append(total_reward)
        self.iterative_metrics['discounted_returns'].append(total_discounted_reward)

        additional_metrics = agent.additional_metrics(trajectory=trajectory,
                                                      num_steps=num_steps,
                                                      logged_episode_values=self.iterative_metrics)
        if additional_metrics and isinstance(additional_metrics, dict):
            for key, value in additional_metrics.items():
                self.iterative_metrics[key].append(value)

    def _sanity_check(self):
        if self.pass_env_to_agent:
            agent_sig = signature(self.agent_class)
            if not agent_sig.parameters.get(self.AGENT_ENV_ARG):
                raise ValueError(f'When pass_env_agent is set to True, \
                                 agent_class needs to have a named argument {self.AGENT_ENV_ARG}')

        if self.iteration_counter not in ['episodes', 'steps']:
            raise ValueError(f'iteration_counter must be either "episodes" or "steps", \
                             received {self.iteration_counter} instead')

        if self.save_iterator not in ['episodes', 'steps']:
            raise ValueError(f'iteration_counter must be either "episodes" or "steps", \
                             received {self.save_iterator} instead')

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


class BaseMetricLogger(abc.ABC):
    def __init__(self):
        self.iterative_metrics = defaultdict(list)

    def save_iterative_metrics(
        self,
        num_steps,
        trajectory,
        total_reward,
        total_discounted_reward,
        agent,
        env,
        *args,
        **kwargs,
    ):
        self.iterative_metrics["returns"].append(total_reward)
        self.iterative_metrics["discounted_returns"].append(total_discounted_reward)

    def reset(self):
        self.iterative_metrics = defaultdict(list)

    def get_metrics(self):
        return {"iterative_metrics": self.iterative_metrics}


class TrainingEngine:
    # Defines the name of the environment argument in the agent class
    # if the agent class is to be initialized with the env
    AGENT_ENV_ARG = "env"

    def __init__(
        self,
        env_class,
        env_parameters,
        agent_class,
        agent_parameters,
        metric_logger,
        discount_rate,
        num_runs,
        num_iterations_per_run,
        iteration_counter="episodes",
        save_frequency=5,
        evaluation_num_iterations=None,
        pass_env_to_agent=False,
        max_steps=None,
    ):
        self.env_class = env_class
        self.env_parameters = env_parameters
        self.agent_class = agent_class
        self.agent_parameters = agent_parameters
        self.metric_logger = metric_logger
        self.discount_rate = discount_rate
        self.num_runs = num_runs
        self.num_iterations_per_run = num_iterations_per_run
        self.iteration_counter = iteration_counter  # episodes or steps
        self.pass_env_to_agent = pass_env_to_agent
        self.save_frequency = save_frequency
        self.max_steps = max_steps

        self.run_metrics = defaultdict(list)

        self._sanity_check()

    def run(self):
        start_time = time.perf_counter()
        self.run_metrics = defaultdict(list)

        for _ in range(self.num_runs):
            self.metric_logger.reset()  # reset for run
            env, agent = self._init_env_agent()

            total_num_steps = [0]
            i_ep = [0]
            counter = i_ep if self.iteration_counter == "episodes" else total_num_steps
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
                    total_discounted_reward += reward * self.discount_rate ** num_steps

                    trajectory.append((prev_state, action, reward))

                    agent.online_update(trajectory=trajectory, num_steps=num_steps)

                    if self.iteration_counter == "steps" and (
                        total_num_steps[0] == 0
                        or (total_num_steps[0] + 1) % self.save_frequency == 0
                    ):
                        self.metric_logger.save_iterative_metrics(
                            num_steps=num_steps,
                            trajectory=trajectory,
                            total_reward=total_reward,
                            total_discounted_reward=total_discounted_reward,
                            agent=agent,
                            env=env,
                        )

                    num_steps += 1
                    total_num_steps[0] += 1
                    if (self.max_steps and num_steps >= self.max_steps) or (
                        counter[0] >= break_value
                    ):
                        break

                agent.update(trajectory=trajectory, num_steps=num_steps)
                print(
                    "ep",
                    i_ep[0],
                    "time",
                    (time.perf_counter() - start_time) / 60,
                    flush=True,
                )

                if self.iteration_counter == "episodes" and (
                    i_ep[0] == 0 or (i_ep[0] + 1) % self.save_frequency == 0
                ):
                    self.metric_logger.save_iterative_metrics(
                        num_steps=num_steps,
                        trajectory=trajectory,
                        total_reward=total_reward,
                        total_discounted_reward=total_discounted_reward,
                        agent=agent,
                        env=env,
                    )
                    # TODO understand the correct interface
                    # run evaluation phase
                    # eval_logged = eval_agent(deepcopy(agent), env, n_test=100, max_steps=100)
                    # eval_state_visitation_entropy = eval_logged["state_visitation_entropy"]
                    # logged_run_values["state_visitation_entropy_eval"].append(eval_state_visitation_entropy)

                i_ep[0] += 1

            iterative_metrics = self.metric_logger.get_metrics()
            for key, value in iterative_metrics.items():
                self.run_metrics[key].append(value)

        return self.run_metrics

    def _init_env_agent(self):
        env = self.env_class(**self.env_parameters)

        if self.pass_env_to_agent:
            self.agent_parameters.update({self.AGENT_ENV_ARG: env})

        agent = self.agent_class(**self.agent_parameters)

        return env, agent

    def _sanity_check(self):
        if self.pass_env_to_agent:
            agent_sig = signature(self.agent_class)
            if not agent_sig.parameters.get(self.AGENT_ENV_ARG):
                raise ValueError(
                    f"When pass_env_agent is set to True, \
                                 agent_class needs to have a named argument {self.AGENT_ENV_ARG}"
                )

        if self.iteration_counter not in ["episodes", "steps"]:
            raise ValueError(
                f'iteration_counter must be either "episodes" or "steps", \
                             received {self.iteration_counter} instead'
            )

    def _external_evaluation(self):
        pass

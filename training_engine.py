import numpy as np
import time


class TrainingEngine:
    def __init__(self, env, agent, training_hyperparams, num_runs, num_episodes_per_run, max_steps=None):
        self.env = env
        self.agent = agent
        self.training_hyperparams = training_hyperparams
        self.num_runs = num_runs
        self.num_episodes_per_run = num_episodes_per_run
        self.max_steps = max_steps

    def run(self):
        # TODO this is a temporary hack!!!
        state_visitation = np.zeros([10, 10], dtype='float')

        start_time = time.perf_counter()
        for i_run in range(self.num_runs):
            logged_run_values = {x: [] for x in ['returns', 'discounted_returns', 'action_entropy_trajectory',
                                                 'state_visitation_entropy_online', 'state_visitation_entropy_eval']}

            # training loop
            state = env.reset()
            for i_ep in range(self.num_episodes_per_run):
                done = False
                trajectory = []
                total_reward = 0
                total_discounted_reward = 0
                steps = 0
                total_entropy = 0

                while not done:
                    prev_state = state

                    # TODO figure out how to obtain state_visitation correctly
                    state_visitation[state[0], state[1]] += 1
                    total_entropy += agent.get_entropy(state)

                    action = agent.get_action(state)
                    state, reward, done = env.step(action)

                    total_reward += reward
                    total_discounted_reward += reward * agent.discount**steps

                    trajectory.append((prev_state, action, reward))

                    agent.online_update(trajectory=trajectory, num_steps_from_start=steps, **self.training_hyperparams)

                    steps += 1
                    if self.max_steps and steps >= self.max_steps:
                        break

                agent.update(trajectory=trajectory, num_steps_from_start=steps, **self.training_hyperparams)

                # reset
                state = env.reset()
                print("ep", i_ep, "time", (time.perf_counter() - start_time) / 60, flush=True)

                # TODO save frequency=5 argument
                if i_ep % 5 == 0:
                    print("ep", i_ep, "time", (time.perf_counter() - start_time) / 60)

                    logged_run_values['returns'].append(total_reward)
                    logged_run_values['discounted_returns'].append(total_discounted_reward)
                    logged_run_values['action_entropy_trajectory'].append(total_entropy/steps)

                    # compute online state visitation
                    state_visitation += 1e-12
                    state_visitation /= np.sum(state_visitation)
                    online_entropy = -np.sum(state_visitation * np.log(state_visitation))
                    logged_run_values["state_visitation_entropy_online"].append(online_entropy)

                    # TODO understand the correct interface
                    # run evaluation phase 
                    # eval_logged = eval_agent(agent, env, n_test=100, max_steps=100)
                    # eval_state_visitation_entropy = eval_logged["state_visitation_entropy"]
                    # logged_run_values["state_visitation_entropy_eval"].append(eval_state_visitation_entropy)

            return logged_run_values


if __name__ == '__main__':

    import SimpleMDP
    import SimpleAgent

    env = SimpleMDP.FourRoomsEnv(extra_wall=False, wall_penalty=False)
    state_visitation = np.zeros([10, 10], dtype='float')

    num_actions = 4
    agent = SimpleAgent.ExAgent(num_actions=num_actions,
                                discount=0.99, baseline_type='minvar', seed=0, env=env,
                                use_natural_pg=False, relative_perturb=False)



    test = TrainingEngine(env=env,
                          agent=agent,
                          agent_hyperparams={'step_size': 0.05, 'perturb': -0.5, 'optimizer': 'SGD', 'discount': 0.99, 'horizon': 200},
                          num_runs=5,
                          num_episodes_per_run=20)
    test.run()
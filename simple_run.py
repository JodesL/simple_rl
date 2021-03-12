#####
# Runs experiments from config file for Compute Canada
# Use CreateJobs.py to make the files first then we pass arguments to this file
#####
import os, sys
import time
import itertools
import numpy as np
import pickle
import collections

import SimpleAgent
import argparse
import SimpleMDP
from Config import config
from cartesian_product import cartesian_product

# def parse_args():
#     parser = argparse.ArgumentParser(description = 'Experiment')
#     # parser.add_argument('--jobs_path', type=str, default='jobs/', help='path to the jobs folder to load the config file')
#     parser.add_argument('--config_path', type=str, help='path to config file to load')
#     parser.add_argument('--alg', type=str, help='name of algorithm')
#     parser.add_argument('--hyp_index', type=int, help='index of the hyperparameter setting')
#     parser.add_argument('--run_index', type=int, default=0, help='index of the run to be done if parallelize_runs=True')

#     return parser.parse_args()

# args = parse_args()
# with open(args.config_path + "config.pkl", 'rb') as f:
#     config = pickle.load(f)

# recreate base save path where config is
# base_save_path = '{}/{}/{}_{}/'.format(config.save_dir, config.date, config.id, config.save_tag)
# os.makedirs(base_save_path, exist_ok=True)

# construct lists of all hyperparameter settings for chosen algorithm
# shared_param_settings = list(itertools.product(*list(config.shared_sweep_params.values())))  # list of hyperparam tuples
# shared_param_names = list(config.shared_sweep_params.keys())
# hyperparameter_settings = list(itertools.product(*list(config.algs_sweep_params[alg].values())))
# hyperparam_names = list(config.algs_sweep_params[alg].keys())
alg = config.algorithms[0]
hyp_index = 1


sweep_params_dict = config.algs_sweep_params[alg]
sweep_params_dict.update(config.shared_sweep_params)
hyperparam_sweep_list = cartesian_product(sweep_params_dict)
hyperparams = hyperparam_sweep_list[hyp_index]  # the hyperparameter setting for the current run

# training
start_time = time.perf_counter()
param_string = ""

for h in hyperparams:
    if h in ('step_size', 'rew_step_size'):
        param_string += h + "{}".format(np.log10(hyperparams[h]))
    else:
        param_string += h + "{}".format(hyperparams[h])
    param_string += '&'
param_string = param_string[:-1]




def eval_agent(agent, env, n_test, max_steps=500):
    # to do separate rollouts to evaluate the agent  at various checkpoints
    disc = 0.99
    returns = []
    disc_returns = []
    env = env.copy()  # make a deep copy

    if env.name=='gridworld':
        state_visitation = np.zeros([5,5], dtype='float')  # we assume we are using the 5x5 gridworld
    elif env.name=='fourrooms':
        state_visitation = np.zeros([10, 10], dtype='float')
    else:
        raise AssertionError("Invalid env", env.name)

    state = env.reset()
    for i_ep in range(n_test):
        #print("ep {}".format(i_ep))
        done = False
        trajectory = []
        steps = 0

        while not done:
            # print(trajectory)
            prev_state = state

            state_visitation[state[0], state[1]] += 1

            action = agent.get_action(state)
            state, reward, done = env.step(int(action))

            #print(prev_state, reward, done, state)
            trajectory.append((prev_state, action, reward))
            steps += 1
            if steps >= max_steps:
                break

        # do updates
        #print(reward, steps)
        ep_rewards = [x[2] for x in trajectory]

        returns.append(np.sum(ep_rewards))
        ep_disc_return = np.sum([ep_rewards[i]* (disc**i) for i in range(len(ep_rewards))])
        disc_returns.append(ep_disc_return)

        # reset
        state = env.reset()
        # state = np.array(env.human_state)

    state_visitation += 1e-12
    state_visitation /= np.sum(state_visitation)
    entropy = -np.sum(state_visitation * np.log(state_visitation))
    results = {"returns": np.array(returns), "disc_returns" : np.array(disc_returns), "state_visitation_entropy": entropy}
    return results


# with open(base_save_path + config.output_file, 'a') as f:
#     print("Algorithm: {}, Env: {}".format()

all_logged_values = {x : [] for x in config.logged_values}

# save_path = base_save_path + "Runs/{}_{}/".format(args.alg, args.hyp_index)  # just use the index instead of the parameter values

# if config.parallelize_runs:
#     save_path += 'run_{}/'.format(args.run_index)

# os.makedirs(save_path, exist_ok=True)
max_steps = hyperparams['horizon']  # I dont expect this to matter but just in case

if config.parallelize_runs:
    num_runs = 1
else:
    num_runs = config.num_runs

for i_run in range(num_runs):
    #print(i_run)
    # initialize environment and agent
    logged_run_values = {x : [] for x in config.logged_values}

    # instantiate environment  TODO we only use one env for now
    if config.environments[0] == 'gridworld':
        env = SimpleMDP.GridWorldEnv(gridsize=config.env_params[0]["size"])  # assume we have only one env
        num_actions = 5
        state_visitation = np.zeros([5, 5], dtype='float')

    elif config.environments[0] == 'binomialtree':
        env = SimpleMDP.BinomialTreeMDP(depth=config.env_params[0]["depth"])
        num_actions = 2
    elif config.environments[0] == 'fourrooms':
        env = SimpleMDP.FourRoomsEnv(extra_wall=config.env_params[0]['extra_wall'], wall_penalty=config.env_params[0]['wall_penalty'])
        num_actions = 4
        state_visitation = np.zeros([10, 10], dtype='float')

    else:
        raise AssertionError("Invalid env name: {}".format(config.environments[0]))

    # TODO change minvar baseline here
    if config.parallelize_runs:
        seed = config.alg_other_params['seed'] + args.run_index
    else:
        seed = config.alg_other_params['seed'] + i_run

    # print(env)

    baseline_type = config.alg_other_params['baseline_type']
    agent = SimpleAgent.PGAgent(num_actions=num_actions,
                                discount=hyperparams['discount'], baseline_type=baseline_type, seed=seed, env=env,
                                use_natural_pg=config.alg_other_params['use_natural_pg'], relative_perturb=config.alg_other_params['relative_perturb'])
    # print(agent.env)

    # training loop
    state = env.reset()
    for i_ep in range(config.num_episodes):
        # with open(base_save_path + config.output_file, 'a') as f:
        #     print("ep {}".format(i_ep), file=f)

        done = False
        trajectory = []
        total_reward = 0
        total_discounted_reward = 0
        steps = 0
        total_entropy = 0

        while not done:
            prev_state = state

            state_visitation[state[0], state[1]] += 1
            total_entropy += agent.get_entropy(state)

            action = agent.get_action(state)
            state, reward, done = env.step(action)

            total_reward += reward
            total_discounted_reward += reward * agent.discount**steps

            trajectory.append((prev_state, action, reward))

            # if doing online updates, do them now
            if alg == 'online_ac_true_q':
                # update only on the most recent transition
                agent.update_ac_true_q([trajectory[-1]], hyperparams['step_size'], hyperparams['perturb'], num_steps_from_start=steps)

            steps += 1
            if steps >= max_steps:
                break

        # do updates
        # could put this all within the agent (but then you have to pass the algorithm)
        if alg == 'reinforce':
            agent.update_reinforce(trajectory, hyperparams['step_size'],  hyperparams['perturb'])#, hyperparams['rew_step_size'])
        elif alg == 'ac_true_q':  # actor-critic solving for the true q-values at each iteration
            # print(agent.env)
            agent.update_ac_true_q(trajectory, hyperparams['step_size'], hyperparams['perturb'])

        # reset
        state = env.reset()
        print("ep", i_ep, "time", (time.perf_counter() - start_time) / 60, flush=True)

        if i_ep % config.save_freq == 0:
            print("ep", i_ep, "time", (time.perf_counter() - start_time) / 60)

            logged_run_values['returns'].append(total_reward)
            logged_run_values['discounted_returns'].append(total_discounted_reward)
            logged_run_values['action_entropy_trajectory'].append(total_entropy/steps)

            # compute online state visitation
            state_visitation += 1e-12
            state_visitation /= np.sum(state_visitation)
            online_entropy = -np.sum(state_visitation * np.log(state_visitation))
            logged_run_values["state_visitation_entropy_online"].append(online_entropy)

            # run evaluation phase
            eval_logged = eval_agent(agent, env, n_test=100, max_steps=100)
            eval_state_visitation_entropy = eval_logged["state_visitation_entropy"]
            logged_run_values["state_visitation_entropy_eval"].append(eval_state_visitation_entropy)

    # with open(base_save_path + config.output_file, 'a') as f:
    # print("run {}".format(i_run), file=f)

    # save the logged values
    for key, value in logged_run_values.items():
        all_logged_values[key].append(value)

# for logged_var, result in all_logged_values.items():
#     all_logged_values[logged_var] = np.array(all_logged_values[logged_var])
#     if logged_var == "returns":
#         # save everything for now
#         np.save(save_path + "all_returns.npy", all_logged_values[logged_var])

#         mean_returns = np.mean(all_logged_values[logged_var], axis=0)
#         std_returns = np.std(all_logged_values[logged_var], axis=0)
#         # np.save(save_path + 'mean_errors.npy', mean_errors)
#         # np.save(save_path + 'std_errors.npy', std_errors)

#         with open(base_save_path + config.output_file, 'a') as f:
#             print("Final: {:.2f} +- {:.2f}, Avg: {:.2f} +- {:.2f}, Time: {:.2f} {} {} Index {}".format(
#                 mean_returns[-1], std_returns[-1], np.mean(mean_returns), np.mean(std_returns), (time.perf_counter() - start_time) / 60,
#                 args.alg, param_string, args.hyp_index), file=f)
#     else:
#         np.save(save_path + logged_var + '.npy', all_logged_values[logged_var])
#     # if logged_var == 'action_entropy_trajectory':
    #     np.save(save_path + "action_entropy_trajectory.npy", all_logged_values[logged_var])
    # if logged_var == 'state_visitation_entropy_online':
    #     np.save(save_path + "state_visitation_entropy_online.npy", all_logged_values[logged_var])
    # if logged_var == 'state_visitation_entropy_eval':
    #     np.save(save_path + "state_visitation_entropy_eval.npy", all_logged_values[logged_var])

# with open(base_save_path + config.output_file, 'a') as f:
#     print("Done! Total runtime (min): {}".format(
#         (time.perf_counter() - start_time)/60), file=f)




###
# Gridworld environment
#
#
###
import time
import numpy as np

import SimpleMDP
from utils import *



class PGAgent:
    def __init__(self, num_actions, discount, baseline_type=None, seed=None, env=None, use_natural_pg=False, relative_perturb=False):
        # tabular
        if env.name.lower() in ['gridworld', 'fourrooms']:
            self.param = np.zeros(shape=[env.gridsize[0], env.gridsize[1], num_actions])  # height x width x num_actions
            self.avg_returns = np.zeros(shape=[env.gridsize[0], env.gridsize[1]])  # used to save avg return from each state (as a baseline)
            self.visit_counts = np.zeros(shape=[env.gridsize[0], env.gridsize[1]])  # used to count visits to each state
        elif env.name.lower() == 'binomialtree':
            # note that the some entries of these arrays are unused, since we only need the lower triangular part to represent the tree
            self.param = np.zeros(shape=[env.depth, env.depth, num_actions])
            self.avg_returns = np.zeros(shape=[env.depth, env.depth])
            self.visit_counts = np.zeros(shape=[env.depth, env.depth])  # used to count visits to each state

        self.seed = seed
        self.num_actions = num_actions
        self.baseline_type = baseline_type  # "avg", "minvar", None
        self.use_natural_pg = use_natural_pg
        self.relative_perturb = relative_perturb  # if True, uses a perturbation which is based on the size of Q(s,a) - b*

        self.discount = discount
        self.rng = np.random.RandomState(seed)
        self.env = env  # this is needed to compute the minimum-variance baseline

        self.q_values = None  # use this to cache previous q-values in ac_true_q
        pass

    def get_policy_prob(self, state, deepcopy=False):
        # returns vector of policy probabilities
        if deepcopy:
            return softmax(self.param[state[0], state[1]]).copy()
        return softmax(self.param[state[0], state[1]])

    def get_action(self, state):
        # returns an action sampled according to the policy probabilities
        # print(self.param[state[0], state[1]])
        return self.rng.choice(np.arange(0, self.num_actions), p=self.get_policy_prob(state))

    def get_entropy(self, state):
        # make sure to be numerically stable
        x = self.param[tuple(state)]
        return softmax_entropy(x)


    def update_reinforce(self, trajectory, step_size, perturb, rew_step_size=None):
        # trajectory is a sequence of transitions for one episode
        # of the form ((s, a, r'), (s', a', r''), ...)
        # note that s must be a tuple (not a numpy array) or else indexing doesn't work properly

        total_reward = 0
        total_discount = self.discount ** (len(trajectory)-1)

        if self.baseline_type == 'minvar':
            # we compute the estimate of the minvar baseline for the initial state only and use it for all the states in a trajectory
            # this corresponds to the minvar baseline of the reinforce estimator on entire trajectories (not the action-dependent one)
            s_init, _, _ = trajectory[0]
            minvar_baseline = self._estimate_minvar_baseline()

        for transition in reversed(trajectory):
            # gradient
            state, a, r = transition
            s = self._state_index(state)
            onehot = np.zeros(self.num_actions)
            onehot[a] = 1

            total_reward = total_reward * self.discount + r

            if self.baseline_type == 'avg':
                baseline = self.avg_returns[s]
                # update the avg returns
                 #1 / (np.sqrt(self.visit_counts[s[0], s[1]]) + 1)  # could change the step size here
                self.avg_returns[s] += rew_step_size * (total_reward - self.avg_returns[s])
            elif self.baseline_type == 'minvar':
                #baseline = self.env.compute_minvar_baseline(s)
                baseline = minvar_baseline
            else:
                baseline = 0
            self.visit_counts[s] += 1


            self.param[s] += step_size * total_discount * ((total_reward - (baseline + perturb)) * (onehot - self.get_policy_prob(s)))
            # note that this previous step has to be done simultaneously for all states for function approx i
            total_discount /= self.discount

        return

    def update_ac_true_q(self, trajectory, step_size, perturb, rew_step_size=None, num_steps_from_start=0, *args, **kwargs):
        # this is the actor-critic update with the true q-values
        # trajectory is a sequence of transitions for one episode
        # of the form ((s, a, r'), (s', a', r''), ...)
        # num_steps is the number of steps since the beginning of the trajectory, this is useful for the online actor-critic
        # note that s must be a tuple (not a numpy array) or else indexing doesn't work properly

        total_reward = 0
        total_discount = self.discount ** ((len(trajectory) - 1) + num_steps_from_start)

        q_values = self._solve_q_values()

        if self.baseline_type == 'minvar':
            # we compute the minvar baseline for all states with the true q-values
            minvar_baselines = self._compute_ac_minvar_baseline(q_values)

        for transition in reversed(trajectory):
            # gradient
            state, a, r = transition
            s = self._state_index(state)
            onehot = np.zeros(self.num_actions)
            onehot[a] = 1

            # total_reward = total_reward * self.discount + r

            if self.baseline_type == 'avg':
                baseline = self.avg_returns[s]
                # update the avg returns
                # 1 / (np.sqrt(self.visit_counts[s[0], s[1]]) + 1)  # could change the step size here
                self.avg_returns[s] += rew_step_size * (total_reward - self.avg_returns[s])
            elif self.baseline_type == 'minvar':
                baseline = minvar_baselines[s]
            else:
                baseline = 0
            self.visit_counts[s] += 1

            if self.use_natural_pg:
                # self.param[s[0], s[1], a] += step_size * total_discount * (q_values[(s[0], s[1], a)] - (baseline + perturb)) / np.clip(self.get_policy_prob(s)[a], 1e-5, 1)
                # is the discount factor correct?
                p = np.clip(self.get_policy_prob(s)[a], 1e-6, 1)
                natural_grad = -np.ones(self.num_actions, dtype='float') / (self.num_actions * p) + 1 / p * onehot
                self.param[s] += step_size * total_discount * (q_values[(s[0], s[1], a)] - (baseline + perturb)) * natural_grad

            else:
                # print("adv", q_values[s[0], s[1], a] - baseline)

                if self.relative_perturb:
                    adv_size = np.max(np.abs(q_values[s[0], s[1]] - baseline))
                    perturb = perturb * adv_size
                self.param[s] += step_size * total_discount * (
                            (q_values[(s[0], s[1], a)] - (baseline + perturb)) * (onehot - self.get_policy_prob(s)))

                # print(onehot - self.get_policy_prob(s))

            # note that this previous step has to be done simultaneously for all states for function approx i

            total_discount /= self.discount

    def _solve_q_values(self):
        # uses dynamic programming to approximately solve for q-values within tolerance
        tolerance = 1e-3

        if self.env.name.lower() == 'fourrooms':
            num_actions = 4
            if self.q_values is None: # initialize array
                self.q_values = 0.5*np.ones((self.env.gridsize[0], self.env.gridsize[1], num_actions))
            q_values = self.q_values  # note this is a reference

            # cache all the possible transitions (assumes self.environment is deterministic)
            transitions = dict()
            for i in range(0, self.env.gridsize[0]):
                for j in range(0, self.env.gridsize[1]):
                    for a in range(0, 4):
                        transitions[(i,j,a)] = self.env._transition(state=(i,j), action=a)
                        # note: transition returns (next_state, reward, done)

            max_change = 999
            while max_change > tolerance:
                # print("change", max_change)
                temp_copy = q_values.copy()
                for i in range(0, self.env.gridsize[0]):
                    for j in range(0, self.env.gridsize[1]):
                        for a in range(0, 4):
                            # print(transitions[(i,j,a)])
                            next_state, reward, done = transitions[(i,j,a)]

                            # next_state = transitions[(i, j, a)]
                            policy_next_state = self.get_policy_prob(next_state)

                            # compute bellman update
                            if done:
                                q_values[(i,j,a)] = reward
                                # print(reward)
                            else:
                                q_values[(i,j,a)] = reward + self.discount * np.sum(q_values[tuple(next_state)] * policy_next_state)

                max_change = np.max(np.abs(q_values - temp_copy))

            return q_values.copy()

        elif self.env.name.lower() == 'gridworld':
            raise AssertionError('Solving q-values Not implemented for gridworld')

    def _compute_ac_minvar_baseline(self, q_values):
        # returns the min-variance baseline for all states
        # requires as input the q-value estimates (could be the true q-values)
        if self.env.name.lower() == 'fourrooms':
            # assumes q_values are given as an array with (state, state, action)
            num_actions = 4
            baselines = np.zeros(tuple(self.env.gridsize), dtype='float')
            for i in range(self.env.gridsize[0]):
                for j in range(self.env.gridsize[1]):
                    policy = self.get_policy_prob([i, j])
                    baselines[(i,j)] = np.sum([self._weight_optimal_baseline(a, policy) * q_values[(i,j,a)] for a in range(num_actions)])
            return baselines

        elif self.env.name.lower() == 'gridworld':
            raise AssertionError('Solving q-values Not implemented for gridworld')

    def _weight_optimal_baseline(self, index, policy_probs):
        if self.use_natural_pg:
            policy_probs = np.clip(policy_probs, 1e-6, 1)
            denom = np.sum([policy_probs[index] / policy_probs])
            weight = 1 / denom
        else:
            denom = 1 - np.sum(np.square(policy_probs))
            num = np.sum(np.square(policy_probs)) - policy_probs[index]**2 + (1-policy_probs[index])**2
            weight = num * policy_probs[index] / denom
        return weight

    def _estimate_minvar_baseline(self, num_rollouts=100, importance_sampling=True):
        # uses rollouts to estmate the minimum-variance baseline
        # use importance sampling for better estimates
        # the behaviour policy is set to be uniform random epsilon of the time or else it picks the same action as
        # the target policy 1-epsilon of the time

        behav_policy_epsilon = 0.1

        disc = 0.99
        disc_returns = []
        gradlogprob_sums = []

        env = self.env.copy()
        if env.name == 'gridworld':
            max_steps = 100
        elif env.name == 'fourrooms':
            max_steps = 200

        state = env.reset()

        for i_ep in range(num_rollouts):
            gradlogprob = np.zeros(self.param.shape)  # stores all the gradients

            ep_rewards= []
            # print("ep {}".format(i_ep))
            done = False
            steps = 0
            while not done:
                # print(trajectory)
                prev_state = state

                action = self.get_action(state)

                # compute gradient log prob for current state and add it to the rest
                s = self._state_index(state)
                onehot = np.zeros(self.num_actions)
                onehot[action] = 1
                gradlogprob[s] += onehot - self.get_policy_prob(s)

                state, reward, done = env.step(int(action))

                ep_rewards.append(reward)

                steps += 1
                if steps >= max_steps:
                    break

            ep_disc_return = np.sum([ep_rewards[i] * (disc ** i) for i in range(len(ep_rewards))])
            disc_returns.append(ep_disc_return)
            gradlogprob_sums.append(np.sum(np.square(gradlogprob)))

            # reset
            state = env.reset()

        # check = np.stack([gradlogprob_sums, disc_returns])
        # print(np.round( check[:, check[0,:].argsort()], 2))

        minvar_baseline = np.sum(np.array(disc_returns) * np.array(gradlogprob_sums)) / np.sum(gradlogprob_sums)
        # print(minvar_baseline)
        return minvar_baseline



    def _state_index(self, state):
        ''' Converts a state into the appropriate indices to index the parameters '''
        if self.env.name.lower() in ('gridworld', 'fourrooms'):
            return state[0], state[1]
        elif self.env.name.lower() == 'binomialtree':
            return state[0], state[1]
        else:
            raise AssertionError("invalid env {}".format(self.env.name))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    np.set_printoptions(suppress=True, linewidth=1000)
    print('Running!')
    # gridsize = (5,5)
    step_size = 0.5
    perturb = -1.5
    # num_steps = 1000
    num_episodes = 500
    disc = 0.99


    env = SimpleMDP.FourRoomsEnv(extra_wall=False, wall_penalty=False)
    # env = SimpleMDP.BinomialTreeMDP(depth=10)
    agent = PGAgent(num_actions=4, baseline_type='minvar', discount=disc, env=env,
                    use_natural_pg=False, relative_perturb=True)

    # rng = np.random.RandomState(3)
    # agent.param = rng.normal(0.0, 1, (10, 10, 4)) # what if we use policies with less entropy?

    # q = agent._solve_q_values()
    # # val = np.mean(q, axis=2)
    # # print(val)
    # b = agent._compute_ac_minvar_baseline(q)
    # print(b)
    # quit()
    def act2str(action):
        if action == 0:
            return '^'
        elif action == 1:
            return 'v'
        elif action == 2:
            return '<'
        elif action == 3:
            return '>'
        else:
            return 'o'

    # agent.param[0,0, 0] = 3
    state = env.reset()

    start_time = time.perf_counter()
    max_steps = 100
    num_steps_per_ep = []
    returns = []
    action_entropy_trajectory = []
    action_entropy_all_states = []

    for i_ep in range(num_episodes):
        done = False
        trajectory = []
        steps = 0

        count_s0 = 0
        s0_policy = []
        s0_param = []
        prev_policy_prob = []

        total_entropy = 0

        while not done:
            # print(trajectory)
            prev_state = state

            # print(agent.get_entropy(state))
            total_entropy += agent.get_entropy(state)

            action = agent.get_action(state)
            state, reward, done = env.step(action)
            # print(prev_state, reward, done, state)
            trajectory.append((prev_state, action, reward))

            prev_policy_prob.append((act2str(action), np.round(agent.get_policy_prob(prev_state, True), 3)))

            # update only on the most recent transition
            # agent.update_ac_true_q([trajectory[-1]], step_size, perturb, num_steps_from_start=0)# steps)


            if np.all(np.equal(np.array([0,0]), prev_state)):
                count_s0 += 1
                s0_policy.append(list(np.round(agent.get_policy_prob(prev_state, True), 3)))
                s0_param.append(list(np.round(agent.param[tuple(prev_state)].copy(), 2)))

            steps += 1
            if steps >= max_steps:
                break
        # log stuff
        for s, a, _ in trajectory:
            prev_policy_prob.append((act2str(a), np.round(agent.get_policy_prob(s), 3) ))

        action_entropy_trajectory.append(total_entropy/steps)

        all_entropies = []
        for i in range(10):
            for j in range(10):
                all_entropies.append(softmax_entropy(agent.get_policy_prob([i,j],True)))
        action_entropy_all_states.append(np.mean(all_entropies))

        # do updates
        # print(prev_policy_prob)
        # print(count_s0, s0_policy)
        # print(count_s0, s0_param)
        # print([a for a, prob in prev_policy_prob])
        print('ep', i_ep, 'rew', reward, round(reward * disc**steps,3), 'steps', steps, 'goal', state)
        num_steps_per_ep.append(steps)

        # agent.update_reinforce(trajectory, step_size, perturb)
        agent.update_ac_true_q(trajectory, step_size, perturb)
        # reset
        state = env.reset()
        # prev_state = None

        returns.append(reward * disc**steps)
        # agent = np.sum([_, _, r for transition in trajectory])

        # check
        current_policy_prob = []
        count_s0 = 0
        for s, a, _ in trajectory:
            # if np.all(np.equal(np.array([0,0]), s)):
            #     count_s0 += 1
                # s0_policy.append(list(np.round(agent.get_policy_prob(s), 3)))

            current_policy_prob.append((act2str(a), np.round(agent.get_policy_prob(s), 3)))

    print(num_steps_per_ep)
    print(np.round(returns, 2))
    print((time.perf_counter() - start_time) / 60, 'min')
    pass


    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)


    import matplotlib.pyplot as plt
    window = 10
    plt.figure()
    plt.plot(running_mean(action_entropy_trajectory, window))
    plt.plot(running_mean(action_entropy_all_states, window))
    plt.ylim(0, 1.4)

    plt.title('baselines{}'.format(perturb))
    plt.figure()
    plt.plot(running_mean(returns, window))
    plt.ylim(0,1)
    plt.title('baselines{}'.format(perturb))


    # agent.param = np.random.normal(0.0, 0.0, (10, 10, 4)) # what if we use policies with less entropy?

    q_values = agent._solve_q_values()

    full_policy = np.zeros([10, 10, 4])
    entropies = np.zeros([10, 10])
    for i in range(10):
        for j in range(10):
            full_policy[i,j] = agent.get_policy_prob((i,j), True)
            entropies[i,j] = softmax_entropy(agent.param[i,j])
            # print(full_policy[i,j])

    v_values = np.sum(q_values * full_policy, axis=2)
    print(np.round(v_values, 3))
    print(np.round(entropies, 3))
    print(np.round(np.max(full_policy, axis=2), 3))
    print(np.argmax(full_policy, axis=2))
    vis_policy = np.argmax(full_policy)

    plt.show()

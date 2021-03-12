import numpy as np
from simple_rl import BaseRLAgent
from activation_funcs import softmax_entropy, softmax


class ReinforceAgent(BaseRLAgent):
    def __init__(self, discount, step_size, perturb, baseline_type, env,
                 rew_step_size=None, use_natural_pg=False, relative_perturb=False, 
                 seed=None):
        super().__init__()
        # setup for fourrooms
        self.env = env
        self.num_actions = 4
        self.param = np.zeros(shape=[env.gridsize[0], env.gridsize[1], self.num_actions])  # height x width x num_actions
        self.avg_returns = np.zeros(shape=[env.gridsize[0], env.gridsize[1]])  # used to save avg return from each state (as a baseline)
        self.visit_counts = np.zeros(shape=[env.gridsize[0], env.gridsize[1]])
        self.state_visitation = np.zeros([10, 10], dtype='float')
        self.total_entropy = 0
        ###

        self.rew_step_size = rew_step_size
        self.step_size = step_size
        self.perturb = perturb
        self.baseline_type = baseline_type
        self.use_natural_pg = use_natural_pg
        self.relative_perturb = relative_perturb

        self.discount = discount
        self.rng = np.random.RandomState(seed)

    def get_action(self, state, *args, **kwargs):
        # returns an action sampled according to the policy probabilities
        # print(self.param[state[0], state[1]])
        return self.rng.choice(np.arange(0, self.num_actions), p=self._get_policy_prob(state))

    def update(self, trajectory, *args, **kwargs):
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
            s = state[0], state[1]
            onehot = np.zeros(self.num_actions)
            onehot[a] = 1

            total_reward = total_reward * self.discount + r

            if self.baseline_type == 'avg':
                baseline = self.avg_returns[s]
                # update the avg returns
                # 1 / (np.sqrt(self.visit_counts[s[0], s[1]]) + 1)  # could change the step size here
                self.avg_returns[s] += self.rew_step_size * (total_reward - self.avg_returns[s])
            elif self.baseline_type == 'minvar':
                # baseline = self.env.compute_minvar_baseline(s)
                baseline = minvar_baseline
            else:
                baseline = 0
            self.visit_counts[s] += 1

            self.param[s] += self.step_size * total_discount * ((total_reward - (baseline + self.perturb)) * (onehot - self._get_policy_prob(s)))
            # note that this previous step has to be done simultaneously for all states for function approx i
            total_discount /= self.discount

        return

    def online_update(self, trajectory, num_steps, *args, **kwargs):
        state = trajectory[-1][0]
        self.state_visitation[state[0], state[1]] += 1
        self.total_entropy += softmax_entropy(self.param[tuple(state)])

    def additional_metrics(self, num_steps, *args, **kwargs):
        self.state_visitation += 1e-12
        self.state_visitation /= np.sum(self.state_visitation)
        online_entropy = -np.sum(self.state_visitation * np.log(self.state_visitation))

        add_metrics = {'action_entropy_trajectory': self.total_entropy / num_steps,
                       'state_visitation_entropy_online': online_entropy}

        return add_metrics

    def _get_policy_prob(self, state, deepcopy=False):
        # returns vector of policy probabilities
        if deepcopy:
            return softmax(self.param[state[0], state[1]]).copy()
        return softmax(self.param[state[0], state[1]])

    def _estimate_minvar_baseline(self, num_rollouts=100, importance_sampling=True):
        # uses rollouts to estmate the minimum-variance baseline
        # use importance sampling for better estimates
        # the behaviour policy is set to be uniform random epsilon of the time or else it picks the same action as
        # the target policy 1-epsilon of the time
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

            ep_rewards = []
            # print("ep {}".format(i_ep))
            done = False
            steps = 0
            while not done:
                # print(trajectory)
                prev_state = state

                action = self.get_action(state)

                # compute gradient log prob for current state and add it to the rest
                s = state[0], state[1]
                onehot = np.zeros(self.num_actions)
                onehot[action] = 1
                gradlogprob[s] += onehot - self._get_policy_prob(s)

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

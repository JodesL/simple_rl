######
# Some simple MDPs to test policy gradient on
# Tree-structured where we can compute the optimal baseline by enumerating all possible paths
#
######
import numpy as np
import itertools
import copy
import math

class BinomialTreeMDP:
    def __init__(self, depth=10):
        ''' This is a recombining binary tree, so for example (left, right) leads to the same node as (right, left) '''
        self.name = 'binomialtree'
        self.pos = None # depth, node
        self.goal_states = [[np.array([depth, 0]), 1.0],
                            [np.array([depth, depth//2]), 0.8]]  # [goal, reward]
        self.depth = depth
        # we put the suboptimal reward in the middle of the

    def reset(self):
        self.pos = np.array([0, 0])
        return self.pos.copy()

    def step(self, action):
        ''' Two possible actions: left (0) or right (1)'''
        if action == 0:
            self.pos[0] += 1
        elif action == 1:
            self.pos[0] += 1
            self.pos[1] += 1
        else:
            raise AssertionError("invalid action".format(action))

        reward = 0
        for goal_state, goal_reward in self.goal_states:
            if np.all(goal_state == self.pos):
               reward = goal_reward

        done = False
        if self.pos[0] == self.depth:
            done = True

        return  self.pos.copy(), reward, done

    def compute_minvar_baseline(self, policy, state):
        ''' Computes the minimum variance baseline for the current state
        policy is a numpy array of shape (depth, depth, num_actions) with the policy probs for each state
        state is a numpy array of the position to compute the baseline for '''
        env_copy = copy.deepcopy(self)  # make a copy so we can use step(...) method

        # enumrate all possible trajactories and compute relevant quantities for the expectations

        # optimal baseline is E[R(\tau) G(\tau)^2] / E[G(\tau)^2] where G(\tau) is the norm of the gradient of the log prob
        numerator = 0
        denominator = 0
        for action_seq in itertools.product([0,1], repeat=self.depth-env_copy.pos[1]):  # remaining actions to do
            env_copy.pos = state.copy()
            s = env_copy.pos

            trajectory_log_prob = 0
            grad_log_prob = 0
            total_reward = 0
            for a in action_seq:
                grad_log_prob += 1 / policy[s[0], s[1], a]
                trajectory_log_prob += math.log(policy[s[0], s[1], a])
                s, r, _ = env_copy.step(a)
                total_reward += r
            numerator += total_reward * grad_log_prob**2 * math.exp(trajectory_log_prob)
            denominator += grad_log_prob**2 * math.exp(trajectory_log_prob)

        return numerator / denominator


    def eval_agent(self, policy):
        ''' Computes the value function for the start state for the given policy
        policy is a numpy array of shape (depth, depth, num_actions) with the policy probs for each state
        Note this assumes that rewards are only found in the leaf nodes but not within the tree  '''

        discount = 1.0  # assumes that no discounting is used (since all rewards are found after the same number of timesteps)
        # use DP
        values = np.zeros(shape=[self.depth+1, self.depth+1])

        for goal_state, reward in self.goal_states:
            values[tuple(goal_state)] = reward

        for i in reversed(range(self.depth)):
            for j in range(i+1):
                values[i,j] = discount * (policy[i,j,0] * values[i+1, j]  + policy[i,j,1] * values[i+1, j+1])

        return values[0,0]

class GridWorldEnv:
    def __init__(self, gridsize=None):
        self.name = 'gridworld'
        # the x-position goes from [0, gridsize[0]-1] and y-position goes from [0, gridsize[1]-1]
        self.pos = None
        self.num_actions = 5
        if gridsize is None:
            gridsize = [5,5]
        self.gridsize = gridsize
        self.steps = np.array([[0, -1], [0, 1], [-1, 0], [1, 0], [0, 0]])
        self.goal_states = [[np.array([self.gridsize[0]-1, self.gridsize[1]-1]), 1], [np.array([self.gridsize[0]-1, 0]), 0.8]]  # [goal, reward]
        self.reset()

        pass

    def reset(self):
        self.pos = np.array([0, 0])
        return self.pos.copy()

    def copy(self):
        copy_env = GridWorldEnv(self.gridsize)
        copy_env.pos = self.pos.copy()
        return copy_env

    def _check_valid_pos(self, state):
        return 0 <= state[0] < self.gridsize[0] and 0 <= state[1] < self.gridsize[1]

    def _check_goal(self, state):
        for goal_state, goal_reward in self.goal_states:
            if np.all(goal_state == state):
                return True, goal_reward
        return False, 0.0

    def step(self, action):
        # if an action brings you outside the grid, don't move
        new_pos = self.pos + self.steps[action]

        if self._check_valid_pos(new_pos):
            self.pos = new_pos

        reached_goal, reward = self._check_goal(self.pos)
        done = False
        if reached_goal:
            done = True

        return self.pos.copy(), reward, done

class FourRoomsEnv:
    def __init__(self, extra_wall=False, wall_penalty=False):
        self.name = 'fourrooms'
        # the x-position goes from [0, gridsize[0]-1] and y-position goes from [0, gridsize[1]-1]
        self.pos = None
        self.gridsize = [10, 10]
        self.num_actions = 4
        self.extra_wall = extra_wall  # adds a wall between the second best goal and the best goal
        self.wall_penalty = wall_penalty  # adds a small negative reward whenever you hit a wall
        self.steps = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])
        self.goal_states = [[np.array([7, 7]), 1.0], [np.array([7, 2]), 0.3], [np.array([2, 7]), 0.6]]  # [goal, reward]
        self.reset()
        pass

    def reset(self):
        self.pos = np.array([0, 0])
        return self.pos.copy()

    def copy(self):
        copy_env = FourRoomsEnv()
        copy_env.pos = self.pos.copy()
        return copy_env

    def _check_valid_pos(self, state):
        return 0 <= state[0] < self.gridsize[0] and 0 <= state[1] < self.gridsize[1]

    def _check_hit_wall(self, state, action):
        # there are walls between states with x-coordinate 4 and 5
        # or y-coordinate 4 and 5
        # the only doorways are in the center of the walls of each room
        if action == 0: # up
            if self.extra_wall:
                # try adding a wall between the second best goal and best
                if state[0] == 5 and not state[1] in [2]:
                    return True
            else:
                if state[0] == 5 and not state[1] in [2, 7]:
                    return True

        elif action == 1: # down
            if self.extra_wall:
                if state[0] == 4 and not state[1] in [2]:
                    return True
            else:
                if state[0] == 4 and not state[1] in [2, 7]:
                    return True
        elif action == 2: # left
            if state[1] == 5 and not state[0] in [2, 7]:
                return True
        elif action == 3: # right
            if state[1] == 4 and not state[0] in [2, 7]:
                return True
        else:
            return False

    def _check_goal(self, state):
        for goal_state, goal_reward in self.goal_states:
            if np.all(goal_state == state):
                return True, goal_reward
        return False, 0.0

    def _transition(self, state, action):
        # returns the reward and next state given a state and action
        # note the transitions are deterministic
        state = np.array(state)
        new_state = state + self.steps[action]
        hit_wall = True

        if self._check_valid_pos(new_state) and not self._check_hit_wall(state, action):
            state = new_state
            hit_wall = False

        reached_goal, reward = self._check_goal(state)
        if self.wall_penalty and hit_wall:
            reward -= 0.01

        done = False
        if reached_goal:
            done = True

        return state, reward, done

    def step(self, action):
        # if an action brings you outside the grid, don't move
        new_pos = self.pos + self.steps[action]

        hit_wall = True
        if self._check_valid_pos(new_pos) and not self._check_hit_wall(self.pos, action):
            self.pos = new_pos
            hit_wall = False

        reached_goal, reward = self._check_goal(self.pos)
        if self.wall_penalty and hit_wall:
            reward -= 0.01

        done = False
        if reached_goal:
            done = True

        return self.pos.copy(), reward, done






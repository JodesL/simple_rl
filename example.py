
from simple_rl import TrainingEngine


import SimpleMDP
from reinforce_agent import ReinforceAgent


test_env = SimpleMDP.FourRoomsEnv(extra_wall=False, wall_penalty=False)

test_agent = ReinforceAgent(discount=0.99,
                            step_size=0.05,
                            perturb=0,
                            baseline_type='minvar',
                            seed=0,
                            rew_step_size=None,
                            env=test_env,
                            use_natural_pg=False,
                            relative_perturb=False)

test = TrainingEngine(env=test_env,
                      agent=test_agent,
                      discount_rate=0.99,
                      num_runs=5,
                      num_episodes_per_run=6,
                      save_frequency=5)
out = test.run()

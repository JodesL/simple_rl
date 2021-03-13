
from simple_rl import TrainingEngine


from SimpleMDP import FourRoomsEnv
from reinforce_agent import ReinforceAgent


agent_parameters = {'discount': 0.99,
                    'step_size': 0.05,
                    'perturb': 0,
                    'baseline_type': 'None',
                    'seed': 0,
                    'rew_step_size': None,
                    'use_natural_pg': False,
                    'relative_perturb': False}

train_engine = TrainingEngine(env_class=FourRoomsEnv,
                              env_parameters={},
                              agent_class=ReinforceAgent,
                              agent_parameters=agent_parameters,
                              pass_env_to_agent=True,
                              discount_rate=0.99,
                              num_runs=1,
                              num_iterations_per_run=300,
                              iteration_counter='steps',
                              save_frequency=50,
                              save_iterator='steps')
out = train_engine.run()
print(out)


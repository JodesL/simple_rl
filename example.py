from simple_rl import TrainingEngine


from SimpleMDP import FourRoomsEnv
from reinforce_agent import ReinforceAgent, ReinforceMetricLogger


agent_parameters = {
    "discount": 0.99,
    "step_size": 0.05,
    "perturb": 0,
    "baseline_type": "None",
    "seed": None,
    "rew_step_size": None,
    "use_natural_pg": False,
    "relative_perturb": False
}

train_engine = TrainingEngine(
    env_class=FourRoomsEnv,
    env_parameters={"extra_wall": False},
    agent_class=ReinforceAgent,
    agent_parameters=agent_parameters,
    metric_logger=ReinforceMetricLogger(discount=0.99),
    pass_env_to_agent=True,
    discount_rate=0.99,
    num_runs=2,
    num_iterations_per_run=300,
    evaluation_num_iterations=100,
    iteration_counter="episodes",
    save_frequency=50
)

out = train_engine.run()
print(out)


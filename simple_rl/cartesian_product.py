import itertools

from simple_rl import TrainingEngine

def cartesian_product(input_dict):
    """Generate cartesian product expansion for hyperparameters

    Args:
        input_dict (dict): dictionary with keys representing hyperparameters and a list of possible values

    Returns:
        list: list of dictionaries representing all possible sets of hyperparameters
    """

    for key, val in input_dict.items():
        if not isinstance(val, list):
            input_dict[key] = [val]

    return [dict(zip(input_dict.keys(), items)) for items in itertools.product(*input_dict.values())]


def hyperparam_sweep(
        agent_parameters_sweep,
        sweep_ids,
        env_class,
        env_parameters,
        agent_class,
        metric_logger,
        discount_rate,
        num_runs,
        num_iterations_per_run,
        iteration_counter="episodes",
        save_frequency=5,
        evaluation_num_episodes=None,
        pass_env_to_agent=False,
        max_steps=None):

    agent_param_sweep_list = cartesian_product(agent_parameters_sweep)
    sweep_ids = list(range(len(agent_param_sweep_list))) if sweep_ids == 'all' else sweep_ids

    for ids in sweep_ids:
        print(f'running id: {ids}')
        print(f'params: {agent_param_sweep_list[ids]}')

        train_engine = TrainingEngine(env_class=env_class,
                                      env_parameters=env_parameters,
                                      agent_class=agent_class,
                                      agent_parameters=agent_param_sweep_list[ids],
                                      metric_logger=metric_logger,
                                      discount_rate=discount_rate,
                                      num_runs=num_runs,
                                      num_iterations_per_run=num_iterations_per_run,
                                      agent_param_id=ids,
                                      iteration_counter=iteration_counter,
                                      save_frequency=save_frequency,
                                      evaluation_num_episodes=evaluation_num_episodes,
                                      pass_env_to_agent=pass_env_to_agent,
                                      max_steps=max_steps)
        train_engine.run()


# Mango: Hierarchical Reinforcement Learning Framework

Mango is a hierarchical reinforcement learning (HRL) framework designed to provide a modular and flexible environment for experimenting with hierarchical decision-making agents. The framework allows the creation of a hierarchy of abstract layers on top of an existing environment, facilitating the learning of more complex behaviors through the decomposition of long term complex action into simpler ones. The related paper is (INSERIRE PAPER DOI)

## Overview

Mango is implemented in Python and built on top of the OpenAI Gym framework. It introduces the concept of abstract layers, each equipped with its own set of abstract actions, policies, and replay memory. The framework enables the training of both lower and higher-level layers, allowing for the development of a hierarchy of decision-making entities.

## Main Features

- **Abstract Layers:** Define a hierarchy layers of abstraction by implementin a class that condensate the information coming from the state of the environment.

- **Compatibility Metrics:** Evaluate the compatibility between abstract transitions in order to give intrinsic rewards according to the abstract actions implemented.



## Usage

### MangoEnv

The `MangoEnv` class serves as the base environment for the Mango framework. It wraps an existing OpenAI Gym environment and provides additional functionalities for hierarchical RL.

```python
# Example usage of MangoEnv
from mango import MangoEnv
from mango.environments import frozen_lake

# Create Frozenlake env 
init_kwargs = dict(
    map_name="RANDOM",
    shape=(16, 16),
    p=0.5,
    goal_pos=None,
    start_pos=None
)
env = frozen_lake.CustomFrozenLakeEnv(**init_kwargs) #type: ignore
env = frozen_lake.ReInitOnReset(env, **init_kwargs)
env = frozen_lake.TensorObservation(env, one_hot=True)

env_shape = (env.unwrapped.ncol, env.unwrapped.nrow)  # type: ignore
obs, _ = env.reset()
print(f"{obs.shape=}, {obs.dtype=}")
plt.imshow(env.render()) #type: ignore
plt.show()


#define the abstractions that characterize the mango environment
from mango.mango import Mango
from mango.actions.abstract_actions import Grid2dMovementOnehot, Grid2dActions

n_layers = 3
gamma = 0.75
mango = Mango(
    environment=env,
    abstract_actions=[
        Grid2dMovementOnehot(
            cell_shape=(2**i, 2**i),
            grid_shape=env_shape,
            agent_channel=0,
            add_valid_channel=False,
            reward=(1 - gamma),
            p_termination=0.1,
        )
        for i in range(1, n_layers)
    ],
    policy_params=dict(
        lr=1e-3,
        gamma=gamma,
        net_params=dict(
            hidden_channels=[4,4,4],
            hidden_features=[64, 16],
            activation_out=torch.nn.Tanh(),
            #residual_connections=True,
            batch_norm=False,
            device=torch.device("cuda") if torch.cuda.is_available() else None,
        ),
    ),
)
print(mango)
```
### Mango

The `Mango` class orchestrates the entire hierarchical RL framework. It manages the base environment, abstract layers, and provides methods for training and exploration.

### Train the options of the Mango Environment

```python
# Use mango_environment for hierarchical RL

N = 10000
#train starting from the lowest layer
for layer in [1,2]:
    #converging epsilon for the training
    for randomness in (pbar := tqdm(np.linspace(1.0, 0.1, N))):
        pbar.set_description(f"randomness: {randomness:.3f}")
        mango.set_randomness(randomness, layer=layer)
        mango.explore(layer=layer, episode_length=1)
        mango.train(layer=layer)
plt.figure(figsize=(12, 6))
plot_loss_reward(mango, Grid2dActions)
frozen_lake.plot_all_qvals(mango, env)
plt.show()
```


## Installation

Install Mango using pip:

```bash
pip install (DA FARE)
```

<!-- 
## Documentation

For detailed documentation, including API reference and examples, please refer to the [documentation](link-to-documentation).

## Contributing

Contributions to Mango are welcome! Please follow the [contribution guidelines](link-to-contributing-guidelines) before submitting a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

---

Feel free to explore and experiment with Mango, and don't hesitate to reach out for any questions or feedback!
-->





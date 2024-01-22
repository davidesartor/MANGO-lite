import frozen_lake
import jax


rng_key = jax.random.PRNGKey(0)
env = frozen_lake.FrozenLake((8, 8), rng_key=rng_key, frozen_prob=0.8)
# env.play()
env.reset()
env.render()

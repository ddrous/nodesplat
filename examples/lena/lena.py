
#%%

# import jax
from nodesplat import *

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
# jax.config.update("jax_enable_x64", True)

# key = jax.random.PRNGKey(42)
key = jax.random.PRNGKey(time.time_ns())
# key = None

ref_image = plt.imread('lena.jpeg')[...,:3]/255.


scene = init_scene(key, ref_image, 1100)
image = render(scene, ref_image)

fig, (ax) = plt.subplots(1, 2)
sbimshow(image, title="Random init", ax=ax[0])
sbimshow(ref_image, title="Reference", ax=ax[1])


nb_iter = 3000
scheduler = optax.exponential_decay(1e-1, nb_iter, 0.85)
optimiser = optax.adam(scheduler)
opt_state = optimiser.init(scene)

losses = []
# start_time = time.time()
for i in tqdm(range(1, nb_iter+1), disable=True):
    scene, opt_state, loss = train_step(scene, ref_image, opt_state, optimiser)
    losses.append(loss)
    if i % 100 == 0 or i <= 3:
        print(f'Iteration: {i}       Loss: {loss:.3f}')
# wall_time = time.time() - start_time

## Number of params in scene
print("\nNumber of params:", jnp.size(scene))
print("Number of pixels:", jnp.size(ref_image))

image = render(scene, ref_image)

fig, (ax) = plt.subplots(1, 2)
sbimshow(image, title="Final render", ax=ax[0])
sbimshow(ref_image, title="Reference", ax=ax[1])


sbplot(losses, title="MAE loss history", y_scale='log', x_label="Iteration");

#%%

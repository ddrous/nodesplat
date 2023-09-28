
#%%

# import jax
from nodesplat import *

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
# jax.config.update("jax_enable_x64", True)

# key = jax.random.PRNGKey(42)
key = jax.random.PRNGKey(time.time_ns())
# key = None


## Example iamges downloaded from https://micro.magnet.fsu.edu/primer/java/digitalimaging/processing/jpegcompression/
ref_image = plt.imread('blood.png')[...,:3]
if ref_image.max() > 1:
    ref_image = ref_image/255.

gaussians = init_gaussians(key, ref_image, 1100)
image = render_image(gaussians, ref_image)

fig, (ax) = plt.subplots(1, 2)
sbimshow(image, title="Random init", ax=ax[0])
sbimshow(ref_image, title="Reference", ax=ax[1])
plt.show()

nb_iter = 3000
scheduler = optax.exponential_decay(1e-1, nb_iter, 0.85)
optimiser = optax.adam(scheduler)
opt_state = optimiser.init(gaussians)

losses = []
# start_time = time.time()
for i in tqdm(range(1, nb_iter+1), disable=True):
    gaussians, opt_state, loss = train_step(gaussians, ref_image, opt_state, optimiser)
    losses.append(loss)
    if i % 100 == 0 or i <= 3:
        print(f'Iteration: {i}       Loss: {loss:.6f}')
# wall_time = time.time() - start_time

## Number of params in scene
print("\nNumber of params:", jnp.size(gaussians))
print("Number of pixels:", jnp.size(ref_image))
print(f"Compression ration: {jnp.size(ref_image)/jnp.size(gaussians):.1f}:1")

image = render_image(gaussians, ref_image)

fig, (ax) = plt.subplots(1, 2)
sbimshow(image, title="Final render", ax=ax[0])
sbimshow(ref_image, title="Reference", ax=ax[1])


sbplot(losses, title="MAE loss history", y_scale='log', x_label="Iteration");

#%%

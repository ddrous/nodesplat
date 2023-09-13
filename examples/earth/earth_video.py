
#%%

# import jax
from nodesplat import *

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
# jax.config.update("jax_enable_x64", True)

key = jax.random.PRNGKey(3875216005)
# key = jax.random.PRNGKey(time.time_ns())
# key = None

ref_video = []
for i in range(1, 25):
    file_number = str(i).zfill(4)
    frame = plt.imread(f'earth_frames/{file_number}.png')[...,:3]/1.
    ref_video.append(frame)
ref_video = jnp.stack(ref_video, axis=0)
nb_frames, width, height, _ = ref_video.shape

model = init_changing_scene(key, ref_video, 500)
transforms = jnp.tile(make_identity_transform(), (nb_frames, 1))
video = render_video(model.scene.gaussians, transforms, ref_video)

def plot_pred_ref_videos(video, ref_video, title="Render"):
    fig, (ax) = plt.subplots(2, 3)
    sbimshow(video[0], title=title+" t=0", ax=ax[0,0])
    sbimshow(video[11], title=title+" t=11", ax=ax[0,1])
    sbimshow(video[23], title=title+" t=23", ax=ax[0,2])

    sbimshow(ref_video[0], title="Ref t=0", ax=ax[1, 0])
    sbimshow(ref_video[11], title="Ref t=11", ax=ax[1, 1])
    sbimshow(ref_video[23], title="Ref t=23", ax=ax[1, 2])
    plt.show()

def plot_pred_video(video, title="Render"):
    fig, (ax) = plt.subplots(1, 3)
    sbimshow(video[0], title=title+" t=0", ax=ax[0])
    sbimshow(video[11], title=title+" t=11", ax=ax[1])
    sbimshow(video[23], title=title+" t=23", ax=ax[2])
    plt.show()


plot_pred_ref_videos(video, ref_video, title="Init")

nb_iter = 1000
scheduler = optax.exponential_decay(1e-1, nb_iter, 0.8)
optimiser = optax.adam(scheduler)

params, static = eqx.partition(model, eqx.is_array)
opt_state = optimiser.init(params)

losses = []
start_time = time.time()
for i in tqdm(range(1, nb_iter+1), disable=True):
    params, opt_state, loss, transforms = train_step_video(params, static, ref_video, opt_state, optimiser)
    losses.append(loss)
    if i % 100 == 0 or i <= 3:
        video = render_video(params.scene.gaussians, transforms, ref_video)
        plot_pred_video(video, title="Iter "+str(i)+" - ")
        print(f'Iteration: {i}        Loss: {loss:.3f}')
wall_time = time.time() - start_time

## Number of params in scene
param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
print("\nNumber of params:", param_count)    ## Count the mlp parms as well
print("Number of pixels:", jnp.size(ref_video))
print("Wall time in h:m:s:", time.strftime("%H:%M:%S", time.gmtime(wall_time)))

model = eqx.combine(params, static)
video = render_video(params.scene.gaussians, transforms, ref_video)

plot_pred_ref_videos(video, ref_video)

sbplot(losses, title="MAE loss history", y_scale='log', x_label="Iteration");

#%%

jnp.sum(params.scene.gaussians[:, 9] > 0.9)

scene_index = jnp.array([0,1,9])
network_input = model.scene.gaussians[:, scene_index].flatten()

angle_trans = model.engine(network_input).squeeze() 
angle_trans

angla = angle_trans[0]%(2*jnp.pi)


# print(key)
# %%

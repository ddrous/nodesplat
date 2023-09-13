
"""
This defines a Scene, which is a collection of 2D Gaussians.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from tqdm import tqdm
import time

from .gaussian import *
from functools import partial

THRESHOLD = 0.9

def render_pixel(scene: Scene2D, x: jnp.ndarray):
    """Render a single pixel."""

    means = scene[:, :2]
    scalings = scene[:, 2:4]
    rotations = scene[:, 4:5]
    colours = scene[:, 5:8]
    opacities = scene[:, 8:]

    densities = jax.vmap(get_density, in_axes=(0, 0, 0, None))(means, scalings, rotations, x)[:, None]
    densities = jnp.nan_to_num(densities, nan=0.0, posinf=0.0, neginf=0.0)

    # return jnp.sum(densities * colours * opacities, axis=0)
    # return jnp.sum(densities * colours, axis=0)
    # return jnp.sum(jax.nn.sigmoid(densities * colours * opacities), axis=0)

    # return jnp.sum(1/ jnp.exp(-densities * colours), axis=0)
    # return jnp.sum(jax.nn.sigmoid(densities * colours), axis=0)
    # return jnp.sum((densities * colours), axis=0)

    # return jnp.clip(jnp.sum(densities * colours, axis=0), 0., 1.)
    # return jnp.clip(jnp.sum(densities * colours * opacities, axis=0), 0., 1.)
    return jnp.clip(jnp.sum(densities * colours, axis=0), 0., 1.)


render_pixels_1D = jax.vmap(render_pixel, in_axes=(None, 0), out_axes=0)
render_pixels_2D = jax.vmap(render_pixels_1D, in_axes=(None, 1), out_axes=1)



def render_image(scene: Scene2D, ref_image: jnp.ndarray):
    """
    Render the scene.
    """

    meshgrid = jnp.meshgrid(jnp.arange(0, ref_image.shape[0]), jnp.arange(0, ref_image.shape[1]))
    pixels = jnp.stack(meshgrid, axis=0).T

    image = render_pixels_2D(scene, pixels)

    # return jnp.nan_to_num(image.squeeze(), nan=0.0, posinf=0.0, neginf=0.0)
    return image.squeeze()


def penalty_loss(image):
    return jnp.mean(jnp.where(image > 1., image, 0.))

def mse_loss(scene: Scene2D, ref_image: jnp.ndarray):
    """Calculate the MSE loss between the rendered image and the reference image."""
    image = render_image(scene, ref_image)
    ## Add penalty for values greater than 1
    return jnp.mean((image - ref_image) ** 2)

def mae_loss(scene: Scene2D, ref_image: jnp.ndarray):
    """Calculate the MSE loss between the rendered image and the reference image."""
    image = render_image(scene, ref_image)
    # return jnp.mean(jnp.abs(image - ref_image)) + 1.*penalty_loss(image)
    return jnp.mean(jnp.abs(image - ref_image))


def dice_loss(scene: Scene2D, ref_image: jnp.ndarray):
    """Calculate the MSE loss between the rendered image and the reference image."""
    image = render_image(scene, ref_image)
    return (jnp.sum(image * ref_image) * 2 / jnp.sum(image) + jnp.sum(ref_image)) + 1.*penalty_loss(image)


@partial(jax.jit, static_argnums=(3,))
def train_step(scene: Scene2D, ref_image: jnp.ndarray, opt_state, optimiser):
    """Perform a single training step."""
    loss, grad = jax.value_and_grad(mae_loss)(scene, ref_image)
    updates, new_opt_state = optimiser.update(grad, opt_state)
    new_scene = optax.apply_updates(scene, updates)
    return new_scene, new_opt_state, loss




















def render_pixel_trans(gaussians: Scene2D, transform: jnp.ndarray, x: jnp.ndarray):
    """Render a single pixel; with some gaussians transformed."""

    means = gaussians[:, :2]
    scalings = gaussians[:, 2:4]
    rotations = gaussians[:, 4:5]
    colours = gaussians[:, 5:8]
    opacities = gaussians[:, 8:9]

    objectness = gaussians[:, 9:]

    nb_gaussians = gaussians.shape[0]
    id_transforms = jnp.tile(make_identity_transform(), (nb_gaussians, 1))
    transforms = jnp.tile(transform, (nb_gaussians, 1))

    transforms  = jnp.where(objectness > THRESHOLD, transforms, id_transforms)        ## TODO the grads wrt gaussians with low objectness should 0

    # jax.debug.print("Transforms: {}", transforms[:])
    # jax.debug.print("Objectness: {}", objectness[:5])

    densities = jax.vmap(get_transformed_density, in_axes=(0, 0, 0, 0, None))(means, scalings, rotations, transforms, x)[:, None]
    densities = jnp.nan_to_num(densities, nan=0.0, posinf=0.0, neginf=0.0)

    return jnp.clip(jnp.mean(densities * colours, axis=0), 0., 1.)


render_pixels_1D_trans = jax.vmap(render_pixel_trans, in_axes=(None, None, 0), out_axes=0)
render_pixels_2D_trans = jax.vmap(render_pixels_1D_trans, in_axes=(None, None, 1), out_axes=1)

def render_video(gaussians: Scene2D, transforms:jnp.ndarray, ref_video: jnp.ndarray):
    """
    Render the video as a series of V images. 
    scene is:        N x 9 gausians, 
    transforms is:   V x 5 x 1 (one transform per frame wrt the previous frame), 
    ref_video is:    V x H x W x 3 
    """

    width, height = ref_video.shape[1], ref_video.shape[2]
    meshgrid = jnp.meshgrid(jnp.arange(0, width), jnp.arange(0, height))
    pixels = jnp.stack(meshgrid, axis=0).T

    ## Compose all the transforms
    cum_transforms = jnp.zeros_like(transforms)
    cum_transforms = cum_transforms.at[0].set(transforms[0])

    # for i in range(1, transforms.shape[0]):
    #     cum_transforms = cum_transforms.at[i].set(compose_transforms(transforms[i], cum_transforms[i-1]))

    ## Redo the above with a jax fori_loop
    def body_fun(i, cum_transforms):
        return cum_transforms.at[i].set(compose_transforms_centered(transforms[i], cum_transforms[i-1]))

    cum_transforms = jax.lax.fori_loop(1, transforms.shape[0], body_fun, cum_transforms)

    video = jax.vmap(render_pixels_2D_trans, in_axes=(None, 0, None))(gaussians, cum_transforms, pixels)

    return video.squeeze()


def mae_loss_video(model: ChangingScene, ref_video: jnp.ndarray):
    """Calculate the MAE loss between the rendered video and the reference video."""

    # scene = params.scene
    gaussians = model.scene.gaussians
    neuralnet = model.engine      ## TODO  pass these two seperately for two different optimisers

    scene_index = jnp.array([0,1,9])    ## TODO We only use center and objecteness for now
    network_input = gaussians[:, scene_index].flatten()

    angle_trans = neuralnet(network_input).squeeze()    ## TODO make the neural net predict smatter things

    objectiveness = gaussians[:, 9:]
    # zero_means = jnp.zeros((gaussians.shape[0], 2))
    changing_means  = jnp.where(objectiveness > THRESHOLD, gaussians[:, :2], 0.)
    changing_count = jnp.sum(jnp.where(objectiveness > 0.5, 1., 0.))

    first_transform = jnp.concatenate((jnp.sum(changing_means, axis=0)/changing_count, angle_trans)).squeeze()

    all_transforms = jnp.tile(first_transform, (ref_video.shape[0], 1))

    video = render_video(gaussians, all_transforms, ref_video)
    return jnp.mean(jnp.abs(video - ref_video)), all_transforms



# @partial(eqx.filter_jit, static_argnums=(3,))
@eqx.filter_jit
def train_step_video(params, static, ref_video: jnp.ndarray, opt_state, optimiser):
    """Perform a single training step."""

    model = eqx.combine(params, static)

    (loss, transforms), grads = eqx.filter_value_and_grad(mae_loss_video, has_aux=True)(model, ref_video)
    
    updates, new_opt_state = optimiser.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, loss, transforms


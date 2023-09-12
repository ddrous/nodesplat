
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


Scene2D = jnp.ndarray


def init_scene(key, image, N: int) -> Scene2D:
    """Returns the initial model params."""
    keys = get_new_keys(key, N)
    gaussians = [init_gaussian(keys[i], image.shape[0], image.shape[1]) for i in range(N)]
    return jnp.stack(gaussians, axis=0)


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



def render(scene: Scene2D, ref_image: jnp.ndarray):
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
    image = render(scene, ref_image)
    ## Add penalty for values greater than 1
    return jnp.mean((image - ref_image) ** 2)

def mae_loss(scene: Scene2D, ref_image: jnp.ndarray):
    """Calculate the MSE loss between the rendered image and the reference image."""
    image = render(scene, ref_image)
    # return jnp.mean(jnp.abs(image - ref_image)) + 1.*penalty_loss(image)
    return jnp.mean(jnp.abs(image - ref_image))


def dice_loss(scene: Scene2D, ref_image: jnp.ndarray):
    """Calculate the MSE loss between the rendered image and the reference image."""
    image = render(scene, ref_image)
    return (jnp.sum(image * ref_image) * 2 / jnp.sum(image) + jnp.sum(ref_image)) + 1.*penalty_loss(image)


@partial(jax.jit, static_argnums=(3,))
def train_step(scene: Scene2D, ref_image: jnp.ndarray, opt_state, optimiser):
    """Perform a single training step."""
    loss, grad = jax.value_and_grad(mae_loss)(scene, ref_image)
    updates, new_opt_state = optimiser.update(grad, opt_state)
    new_scene = optax.apply_updates(scene, updates)
    return new_scene, new_opt_state, loss


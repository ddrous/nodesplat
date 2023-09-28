"""
This defines the 2D Gaussian model.
"""
import jax
import jax.numpy as jnp
from .utils import get_new_keys

from typing import NamedTuple
import equinox as eqx



class ChangingScene(NamedTuple):    ## The gaussians now have objectness, and some of them are transformed by the neural net
  scene: eqx.Module
  engine: eqx.Module



def init_gaussian(key, width=256., height=256.) -> jnp.ndarray:
    """Returns the initial model params."""
    keys = get_new_keys(key, 6)

    ## Uniformly initialise parameters of a 2D gaussian
    mean = jax.random.uniform(keys[0], (2,), minval=0, maxval=min(width, height))
    scaling = jax.random.uniform(keys[1], (2,), minval=0, maxval=min(width, height)/10)
    rotation = jax.random.uniform(keys[2], (1,), minval=0, maxval=2*jnp.pi)
    colour = jax.random.uniform(keys[3], (3,), minval=0, maxval=1)
    opacity = jax.random.uniform(keys[4], (1,), minval=0, maxval=1)

    objectness = jax.random.uniform(keys[5], (1,), minval=0, maxval=1)

    return jnp.concatenate([mean, scaling, rotation, colour, opacity, objectness])



def init_gaussians(key, image, N: int) -> jnp.ndarray:
    """Returns the initial model params."""
    keys = get_new_keys(key, N)
    gaussians = [init_gaussian(keys[i], image.shape[0], image.shape[1]) for i in range(N)]
    return jnp.stack(gaussians, axis=0)


class EqxScene(eqx.Module):
    gaussians: jnp.ndarray
    def __init__(self, key, image, N):
        self.gaussians = init_changing_scene(key, image, N)


# def init_static_scene(key, image, N: int) -> eqx.Module:
#     """Returns the initial model params."""
#     key = get_new_keys(key, 1)
#     scene = EqxScene(key, image, N)

#     return scene



def init_changing_scene(key, video, N: int) -> ChangingScene:
    """Returns the initial model params."""
    keys = get_new_keys(key, 2)
    scene = EqxScene(keys[0], video[0], N)

    ## TODO width_size should be decreasing [N, 100, 5]
    mlp = eqx.nn.MLP(in_size=3*N, out_size=3, width_size=100, depth=2, activation=jax.nn.tanh, key=keys[1])

    return ChangingScene(scene, mlp)




def make_rotation_matrix(angle):
    cos, sin = jnp.cos(angle), jnp.sin(angle)
    return jnp.array([[cos, -sin], [sin, cos]]).squeeze()

def get_covariance(scaling, rotation_angle):
    """Calculate the covariance matrix. """
    scaling_matrix = jnp.diag(scaling)
    rotation_matrix = make_rotation_matrix(rotation_angle)

    covariance =  rotation_matrix @ scaling_matrix @ scaling_matrix.T @ rotation_matrix.T 

    # jax.debug.print("Is positive semi-definite: {}", is_positive_semi_definite(covariance))

    # import jax.numpy as jnp
    # jax.debug.breakpoint()
    return covariance


def get_density(mean, scaling, rotation, x):
    """Calculate the density of the gaussian at a given point."""

    x_ = (x - mean)[:, None]

    res =  jnp.exp(-0.5 * x_.T @ jnp.linalg.inv(get_covariance(scaling, rotation)) @ x_).squeeze()

    ## Nan to Num
    # return jnp.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
    return res













def transform_point(transform, x):
    """Transform a point using the given transform (translation + rotation)."""
    t_center = transform[:2]
    t_angle = transform[2]
    t_translation_vec = transform[3:5]
    t_rotation_mat = make_rotation_matrix(t_angle)

    return t_rotation_mat@(x - t_center + t_translation_vec) + t_center

def transform_point_inverse(transform, x_prime):
    """Inverse transform a point x_prime into x given the transform"""
    t_center = transform[:2]
    t_angle = transform[2]
    t_translation_vec = transform[3:5]
    t_rotation_mat_inv = jnp.linalg.inv(make_rotation_matrix(t_angle))

    return t_rotation_mat_inv@(x_prime - t_center) + t_center - t_translation_vec


def make_identity_transform():
    """Make an identity transform."""
    return jnp.array([0., 0., 0., 0., 0.])


def get_transformed_density(mean, scaling, rotation, transform, x):
    """Calculate the density of the gaussian at a given point."""

    x_inv = transform_point_inverse(transform, x)

    return get_density(mean, scaling, rotation, x_inv)


def compose_transforms(transform1, transform2):
    """Compose two transforms."""
    t1_center = transform1[:2]
    t1_angle = transform1[2]
    t1_translation_vec = transform1[3:5]
    t1_rotation_mat = make_rotation_matrix(t1_angle)

    t2_center = transform2[:2]
    t2_angle = transform2[2]
    t2_translation_vec = transform2[3:5]
    t2_rotation_mat = make_rotation_matrix(t2_angle)

    t_angle = t1_angle + t2_angle
    t_translation_vec = t1_translation_vec + t2_translation_vec
    t_rotation_mat = t1_rotation_mat@t2_rotation_mat
    id_mat = jnp.diag(jnp.ones_like(t_translation_vec))

    # t_center = jnp.linalg.inv(t_rotation_mat-id_mat)@(t_rotation_mat@(t2_translation_vec+t1_center) - t2_rotation_mat@(t2_translation_vec+t1_center-t2_center)-t2_center)

    ## Solve a linear system instead of inverting the matrix
    t_center = jnp.linalg.solve(t_rotation_mat-id_mat, t_rotation_mat@(t2_translation_vec+t1_center) - t2_rotation_mat@(t2_translation_vec+t1_center-t2_center)-t2_center)
    t_center = jnp.nan_to_num(t_center, nan=0.0, posinf=0.0, neginf=0.0)

    return jnp.concatenate([t_center, jnp.array([t_angle]), t_translation_vec])


def compose_transforms_centered(transform1, transform2):
    """Compose two transforms. The center of the second transform is the image of the center of the first"""
    t2_center = transform_point(transform1, transform1[:2])
    new_transform2 = jnp.concatenate([t2_center, transform2[2:]])

    return compose_transforms(transform1, new_transform2)

def is_positive_semi_definite(matrix):
    # import numpy as np
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False
    
    # Compute eigenvalues
    eigenvalues, _ = jnp.linalg.eigh(matrix)
    # print(eigenvalues)

    # Check if all eigenvalues are non-negative
    return (eigenvalues[:] >= 0).all()
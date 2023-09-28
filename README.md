# NodeSplat [LOGO]

## Features

NodeSplat provides a way to represent images and videos as gaussians rather than pixel values. This has three major advantages:
- **Reduced basis approximation**: the number of learnable parameters for an image or a video is often times orders of magnitude smaller than its number of pixels. Upon training, having such a compressed representation of our images might revolutionize the image/video compression industry [IMAGES](Let's compare our compression ratio to PNG and JPEG !). _Can we make the gaussian format scalable like SVG ?_
- **Static video generation**: naively learning a video with a moving ojects inside it amounts to learning the static components in the video (the background), usefull for a miriad of tasks. By using gausians and framing the task as a minimisation problem, our method is much faster than alternative strategies for extracting static backgrounds from videos [IMAGES](What are these strategies ?)
- **Robust physics engine**: Upon capturing the static background of a video, one can add additional gaussians for each moving object in the foreground and track their instatanuous change in properties (**color**, mean, etc.) accros frames as neural ODE [GIF](Gif showing how the color changes accross frames. This should be COOOOL !). Moreover, learning the physics enables user interactivng with the foreground objects. With a single click of the mouse, users can modify trajectories of objects improving on [GENERATIVE IMAGE DYNAMICS][https://generative-dynamics.github.io/]

NodeSPlat is heavily inspired by the idea of [3D Gaussian Splatting](), from which we improve by emphasing the three points above. The second big inspiration is the idea of representing scenes as mesh-free object, i.e. the fundamental elements (the gaussians) are not connected in a structured way like pixels are. This mesh-free and order-less representation is usefull for several tasks in computer graphics and scientific computing [LINK TO UPDEC].   

Additionally, NodeSplat is fully compatible with any Jax program. Using the proviged `Scene` classes (NamedTuples) and their corresponding renderers, one can quickly plug these as drop-in replacements for any image or video generator in your system. This tool uses three of Jax's major transformations: `jit`, `vmap`, and `grad` (ADD `pmap` in the future); as such, reproducing the tasks above represents an excellent learning task for anyone looking to get into Jax for machine learning.

## Getting started
`pip install nodesplat`

## Example usage

### Image generation
```python
import nodesplat as nds

## A quick program that reconstructs each pixel with a loss function

scene = Scene2D
rendered_image =

## use my scene and rendered image in the loss function


```

### Learning physics
```python
## Something that learns physics
```
See the example notebooks for help on this !



## Resources
- Jax
- Equinox
- Optax


## Citing us

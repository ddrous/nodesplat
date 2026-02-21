#%% Cell 1: Imports and Configuration
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
from jax.flatten_util import ravel_pytree

import seaborn as sns
sns.set(style="white", context="talk")

# --- Configuration ---
SEED = 2026
key = jax.random.PRNGKey(SEED)

NB_EPOCHS = 25
NB_ITER_PER_EPOCH = 300
PRINT_EVERY = 10
LEARNING_RATE = 1e-4
P_FORCING = 0.0
REC_FEAT_DIM = 1024

ROOT_WIDTH = 32
ROOT_DEPTH = 2
NUM_FOURIER_FREQS = 16

# --- Chunking & Rollout Configuration ---
CHUNK_SIZE = 12
ROLLOUT_STEPS = 36

# --- NEW: Inner Optimization Configuration ---
INNER_GD_STEPS = 64     # Number of SGD steps to overfit the first frame
INNER_GD_LR = 1e-2      # Learning rate for the inner optimization loop

#%% Cell 2: Plotting Helpers & Data Loading
def sbimshow(img, title="", ax=None):
    img = np.clip(img, 0.0, 1.0)
    if ax is None:
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    else:
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

def plot_pred_ref_videos(video, ref_video, title="Render"):
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    t_mid = len(video) // 2
    t_end = len(video) - 1
    
    sbimshow(video[0], title=f"{title} t=0", ax=ax[0, 0])
    sbimshow(video[t_mid], title=f"{title} t={t_mid}", ax=ax[0, 1])
    sbimshow(video[t_end], title=f"{title} t={t_end}", ax=ax[0, 2])

    sbimshow(ref_video[0], title="Ref t=0", ax=ax[1, 0])
    sbimshow(ref_video[t_mid], title=f"Ref t={t_mid}", ax=ax[1, 1])
    sbimshow(ref_video[t_end], title=f"Ref t={t_end}", ax=ax[1, 2])
    plt.tight_layout()
    plt.show()

def plot_pred_video(video, title="Render"):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    t_mid = len(video) // 2
    t_end = len(video) - 1
    
    sbimshow(video[0], title=f"{title} t=0", ax=ax[0])
    sbimshow(video[t_mid], title=f"{title} t={t_mid}", ax=ax[1])
    sbimshow(video[t_end], title=f"{title} t={t_end}", ax=ax[2])
    plt.tight_layout()
    plt.show()

def plot_pred_ref_videos_rollout(video, ref_video, title="Rollout Render"):
    """Specialized plotting function for the extended rollout sequence."""
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    
    t_mid = 23 # End of the original available data
    t_end = len(video) - 1 # End of the forecast (t=47)

    sbimshow(video[0], title=f"{title} t=0", ax=ax[0, 0])
    sbimshow(video[t_mid], title=f"{title} t={t_mid} (End Ref)", ax=ax[0, 1])
    sbimshow(video[t_end], title=f"{title} t={t_end} (Forecast)", ax=ax[0, 2])

    sbimshow(ref_video[0], title="Ref t=0", ax=ax[1, 0])
    sbimshow(ref_video[t_mid], title=f"Ref t={t_mid}", ax=ax[1, 1])
    sbimshow(ref_video[t_end], title=f"Ref t={t_end} (Zeroes)", ax=ax[1, 2])
    plt.tight_layout()
    plt.show()

# --- Data Loading ---
print("Loading Earth frames...")
ref_video = []
try:
    for i in range(1, 25):
        file_number = str(i).zfill(4)
        frame = plt.imread(f'earth_frames/{file_number}.png')[..., :3] / 1.
        ref_video.append(frame)
    ref_video = jnp.stack(ref_video, axis=0)
except FileNotFoundError:
    print("earth_frames folder not found. Generating dummy sequence for testing...")
    ref_video = jax.random.uniform(key, (24, 64, 64, 3))

nb_frames, H, W, C = ref_video.shape
print(f"Video shape: {ref_video.shape}")

# Precompute Normalized Coordinates
y_coords = jnp.linspace(-1, 1, H)
x_coords = jnp.linspace(-1, 1, W)
X_grid, Y_grid = jnp.meshgrid(x_coords, y_coords)
coords_grid = jnp.stack([X_grid, Y_grid], axis=-1)  # [H, W, 2]


#%% Cell 3: Model Definition
def fourier_encode(x, num_freqs):
    freqs = 2.0 ** jnp.arange(num_freqs)
    angles = x[..., None] * freqs[None, None, :] * jnp.pi
    angles = angles.reshape(*x.shape[:-1], -1)
    return jnp.concatenate([x, jnp.sin(angles), jnp.cos(angles)], axis=-1)

class RootMLP(eqx.Module):
    layers: list

    def __init__(self, in_size, out_size, width, depth, key):
        keys = jax.random.split(key, depth + 1)
        self.layers = [eqx.nn.Linear(in_size, width, key=keys[0])]
        for i in range(depth - 1):
            self.layers.append(eqx.nn.Linear(width, width, key=keys[i+1]))
        self.layers.append(eqx.nn.Linear(width, out_size, key=keys[-1]))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)

class CNNEncoder(eqx.Module):
    """
    A simple CNN that downsamples spatial dimensions and applies a dense layer.
    Used for the recurrence feature extractor \psi.
    """
    layers: list
    
    def __init__(self, in_channels, out_dim, spatial_shape, key, hidden_width=16, depth=3):
        H, W = spatial_shape
        keys = jax.random.split(key, depth + 1)
        
        conv_layers = []
        current_in_channels = in_channels
        current_out_channels = hidden_width
        
        for i in range(depth):
            conv_layers.append(
                eqx.nn.Conv2d(current_in_channels, current_out_channels, kernel_size=3, stride=2, padding=1, key=keys[i])
            )
            current_in_channels = current_out_channels
            current_out_channels *= 2
            
        dummy_x = jnp.zeros((in_channels, H, W))
        for layer in conv_layers:
            dummy_x = layer(dummy_x)
            
        flat_dim = dummy_x.reshape(-1).shape[0]
        
        self.layers = conv_layers + [eqx.nn.Linear(flat_dim, out_dim, key=keys[depth])]
        
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = jax.nn.relu(x)
        x = x.reshape(-1)
        x = self.layers[-1](x)
        return x


class WARP(eqx.Module):
    A: jax.Array
    B: jax.Array
    theta_base: jax.Array  # NEW: Learned prior for the inner optimization
    controlnet_psi: CNNEncoder
    hypernnet_phi: CNNEncoder
    
    root_structure: RootMLP = eqx.field(static=True)
    unravel_fn: callable = eqx.field(static=True)
    d_theta: int = eqx.field(static=True)
    num_freqs: int = eqx.field(static=True)
    frame_shape: tuple = eqx.field(static=True)

    def __init__(self, root_width, root_depth, num_freqs, frame_shape, key):
        k_root, k_A, k_B, k_psi = jax.random.split(key, 4)
        self.num_freqs = num_freqs
        self.frame_shape = frame_shape
        H, W, C = frame_shape
        
        coord_dim = 2 + 2 * 2 * num_freqs 
        template_root = RootMLP(coord_dim, 7, root_width, root_depth, k_root)

        flat_params, self.unravel_fn = ravel_pytree(template_root)
        self.d_theta = flat_params.shape[0]
        self.root_structure = template_root
        
        # --- NEW: Replaced hypernet with a learnable base state ---
        self.theta_base = flat_params
        self.hypernnet_phi = CNNEncoder(in_channels=1 * C, out_dim=self.d_theta, spatial_shape=(H, W), key=k_B, hidden_width=16, depth=3)
        
        d_feat = REC_FEAT_DIM
        self.controlnet_psi = CNNEncoder(in_channels=1 * C, out_dim=d_feat, spatial_shape=(H, W), key=k_psi, hidden_width=16, depth=3)
        
        self.A = jnp.eye(self.d_theta)
        self.B = jnp.zeros((self.d_theta, d_feat))

        print(f"Model Initialized:")
        print(f"  d_theta (State Size): {self.d_theta}")
        print(f"  Matrix A Shape: {self.A.shape}")
        print(f"  Matrix B Shape: {self.B.shape}")

    def render_pixels(self, thetas, coords):
        def render_pt(theta, coord):
            root = self.unravel_fn(theta)
            encoded_coord = fourier_encode(coord, self.num_freqs)
            out = root(encoded_coord)
            
            rgb_fg = jax.nn.sigmoid(out[0:3])
            rgb_bg = jax.nn.sigmoid(out[3:6])
            alpha  = jax.nn.sigmoid(out[6:7])
            return alpha * rgb_fg + (1.0 - alpha) * rgb_bg

        return jax.vmap(render_pt)(thetas, coords)

    def optimize_theta0(self, init_gt_frame, flat_coords):
        """
        Inner optimization loop: Overfits theta to the initial frame.
        """
        H, W, C = self.frame_shape

        def inner_loss_fn(theta):
            thetas_frame = jnp.tile(theta, (H*W, 1))
            pred_flat = self.render_pixels(thetas_frame, flat_coords)
            pred_frame = pred_flat.reshape(H, W, C)
            return jnp.mean(jnp.abs(pred_frame - init_gt_frame))

        # We need the gradient of the inner loss with respect to theta
        grad_fn = jax.grad(inner_loss_fn)

        def sgd_step(theta, _):
            grads = grad_fn(theta)
            # Simple Gradient Descent step
            theta_next = theta - INNER_GD_LR * grads
            return theta_next, None

        # Run the inner optimization loop starting from our learned prior
        theta_opt, _ = jax.lax.scan(sgd_step, self.theta_base, None, length=INNER_GD_STEPS)
        return theta_opt

    def get_thetas_and_preds(self, ref_video, p_forcing, key, coords_grid):
        H, W, C = self.frame_shape
        flat_coords = coords_grid.reshape(-1, 2)
        
        # --- NEW: Run inner optimization to get theta_0 ---
        init_gt_frame = ref_video[0]
        # theta_0 = self.optimize_theta0(init_gt_frame, flat_coords)
        # theta_0 = jax.lax.stop_gradient(self.optimize_theta0(init_gt_frame, flat_coords))
        # theta_0 = self.theta_base

        # Get theta_0 directly from the hypernetwork for faster execution (no inner loop)
        init_frame_feats = self.hypernnet_phi(jnp.transpose(init_gt_frame, (2, 0, 1)))
        # theta_0 = self.theta_base + init_frame_feats
        theta_0 = init_frame_feats

        # init_frame_mu_sigma = self.hypernnet_phi(jnp.transpose(init_gt_frame, (2, 0, 1)))
        # ## Sample from a Gaussian distribution for theta_0 to introduce stochasticity in the initial state
        # init_frame_mu, init_frame_sigma = jnp.split(init_frame_mu_sigma, 2, axis=-1)
        # init_frame_feats = init_frame_mu + init_frame_sigma * jax.random.normal(key, init_frame_mu.shape)
        # # theta_0 = self.theta_base + init_frame_feats
        # theta_0 = init_frame_feats

        def scan_step(state, gt_curr_frame):
            theta, prev_frame_selected, k = state
            k, subk = jax.random.split(k)
            
            thetas_frame = jnp.tile(theta, (H*W, 1))
            pred_flat = self.render_pixels(thetas_frame, flat_coords)
            pred_frame = pred_flat.reshape(H, W, C)
            
            use_gt = jax.random.bernoulli(subk, p_forcing)
            frame_t = jnp.where(use_gt, gt_curr_frame, pred_frame)
            
            frame_t_feats = self.controlnet_psi(jnp.transpose(frame_t, (2, 0, 1)))
            prev_frame_selected_feats = self.controlnet_psi(jnp.transpose(prev_frame_selected, (2, 0, 1)))

            dx_feat = (frame_t_feats - prev_frame_selected_feats) / jnp.sqrt(frame_t_feats.size)
            theta_next = self.A @ theta + self.B @ dx_feat
            
            new_state = (theta_next, frame_t, subk)
            return new_state, pred_frame
            
        # init_frame = jnp.zeros((H, W, C))
        # init_state = (theta_0, init_frame, key)
        # _, pred_video = jax.lax.scan(scan_step, init_state, ref_video)

        init_frame = ref_video[0]     ##TODO use the future, like LAM
        init_state = (theta_0, init_frame, key)
        ref_video = jnp.concatenate((ref_video[1:], ref_video[-1:]), axis=0)
        _, pred_video = jax.lax.scan(scan_step, init_state, ref_video)

        return pred_video

#%% Cell 4: Initialization & Training
def count_trainable_params(model):
    def count_params(x):
        if isinstance(x, jnp.ndarray) and x.dtype in [jnp.float32, jnp.float64]: return x.size
        else: return 0
    param_counts = jax.tree_util.tree_map(count_params, model)
    return sum(jax.tree_util.tree_leaves(param_counts))

key, subkey = jax.random.split(key)
model = WARP(ROOT_WIDTH, ROOT_DEPTH, NUM_FOURIER_FREQS, (H, W, C), subkey)
A_init = model.A.copy()

print(f"Total Trainable Parameters in WARP: {count_trainable_params(model)}")

scheduler = optax.exponential_decay(LEARNING_RATE, transition_steps=NB_ITER_PER_EPOCH*NB_EPOCHS, decay_rate=0.1)
# scheduler = LEARNING_RATE
optimizer = optax.adam(scheduler)
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

# The train_step now compiles to expect a chunked sequence
@eqx.filter_jit
def train_step(model, opt_state, key, ref_video_chunk, coords_grid, p_forcing):
    # print("Chcunk shape in train_step:", ref_video_chunk.shape)  # Debug print to confirm chunk shape   
    # def loss_fn(m):
    #     pred_video = m.get_thetas_and_preds(ref_video_chunk, p_forcing, key, coords_grid)
    #     loss_full = jnp.mean(jnp.abs(pred_video[1:] - ref_video_chunk[1:]))
    #     loss_t0 = jnp.mean(jnp.abs(pred_video[0] - ref_video_chunk[0]))
    #     # return loss_full + 1.0 * loss_t0

    #     return loss_t0

    def loss_fn(m):
        # 1. Standard Forward Pass
        pred_video = m.get_thetas_and_preds(ref_video_chunk, p_forcing, key, coords_grid)
        
        # loss_full = jnp.mean(jnp.abs(pred_video[1:] - ref_video_chunk[1:]))
        # loss_t0 = jnp.mean(jnp.abs(pred_video[0] - ref_video_chunk[0]))
        # return loss_full + 1.0 * loss_t0
        
        loss_full = jnp.mean((pred_video[:] - ref_video_chunk[:])**2)

        # --- AUXILIARY AUTOENCODER REGULARIZATION ---
        # Pick a random frame from the chunk
        k_rand, _ = jax.random.split(key)
        rand_idx = jax.random.randint(k_rand, (), 0, ref_video_chunk.shape[0])
        rand_gt_frame = ref_video_chunk[rand_idx]
        
        # Route it directly through the initialization module
        # theta_rand = m.optimize_theta0(rand_gt_frame, coords_grid.reshape(-1, 2))
        theta_rand = m.hypernnet_phi(jnp.transpose(rand_gt_frame, (2, 0, 1)))
        
        # ## Add noise to theta_rand to encourage robustness (optional)
        # noise = 0.01 * jax.random.normal(key, theta_rand.shape)
        # theta_rand = theta_rand + noise

        # Render the randomly initialized state
        H, W, C = ref_video_chunk.shape[1:]
        thetas_frame_rand = jnp.tile(theta_rand, (H*W, 1))
        pred_flat_rand = m.render_pixels(thetas_frame_rand, coords_grid.reshape(-1, 2))
        pred_frame_rand = pred_flat_rand.reshape(H, W, C)
        
        # Apply the motion weight to the autoencoder loss as well
        # loss_ae = jnp.mean(jnp.abs(pred_frame_rand - rand_gt_frame))
        
        ## Use other losses like perceptual loss or feature matching loss for better gradients (optional)
        loss_ae = jnp.mean((pred_frame_rand - rand_gt_frame)**2)

        # return loss_full + loss_t0 + loss_ae
        return loss_full + loss_ae
        # return loss_t0 + loss_ae
        # return loss_ae

    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val

all_losses = []
start_time = time.time()

for epoch in range(NB_EPOCHS):
    print(f"\nEPOCH: {epoch+1}")
    epoch_losses = []
    
    pbar = tqdm(range(NB_ITER_PER_EPOCH))
    for i in pbar:
        key, subkey, k_idx = jax.random.split(key, 3)
        
        # --- NEW: Random Chunking Logic ---
        # Pick a random starting index ensuring we have CHUNK_SIZE frames available
        start_idx = jax.random.randint(k_idx, (), minval=0, maxval=nb_frames - CHUNK_SIZE + 1)
        # start_idx = 12
        
        ## Make sure the chunk_size is in [CHUNK_SIZE=12, 16, 20, 24] to add some variability in training
        # chunk_size = jax.random.choice(k_idx, jnp.array([CHUNK_SIZE, 16, 20, 24]))
        # # chunk_size = CHUNK_SIZE
        # # chunk_size = jnp.minimum(chunk_size, nb_frames - start_idx)
        # chunk_size = jnp.maximim(chunk_size, nb_frames - start_idx)

        # start_idx = jax.random.randint(k_idx, (), minval=0, maxval=nb_frames - 2 + 1)
        # chunk_size = jax.random.randint(k_idx, (), minval=2, maxval=nb_frames - start_idx + 1)
        # ### chunk_size = (chunk_size // 2) * 2

        ## We want to pick a random start index between that will allow us to 
        min_chunk_size, max_chunk_size = 4, nb_frames
        possible_chunk_sizes = jnp.arange(min_chunk_size, max_chunk_size + 1, 4)  # [4, 8, 12, 16, 20, 24]
        chunk_size = jax.random.choice(k_idx, possible_chunk_sizes)
        start_idx = jax.random.randint(k_idx, (), minval=0, maxval=nb_frames - chunk_size + 1)

        # chunk_size = CHUNK_SIZE
        # Slice the 4 frames dynamically
        ref_video_chunk = jax.lax.dynamic_slice(
            ref_video, 
            start_indices=(start_idx, 0, 0, 0), 
            slice_sizes=(chunk_size, H, W, C)
        )

        model, opt_state, loss = train_step(model, opt_state, subkey, ref_video_chunk, coords_grid, P_FORCING)
        epoch_losses.append(loss)
        
        if i % PRINT_EVERY == 0:
            pbar.set_description(f"Loss: {loss:.4f} | start_idx: {start_idx}")
            
    all_losses.extend(epoch_losses)
    
    # Render progress at end of epoch using the FULL video sequence (to check generalization)
    current_video = eqx.filter_jit(model.get_thetas_and_preds)(ref_video, P_FORCING, key, coords_grid)
    plot_pred_video(current_video, title=f"Epoch {epoch+1} Render")

wall_time = time.time() - start_time
print("\nWall time for WARP training in h:m:s:", time.strftime("%H:%M:%S", time.gmtime(wall_time)))

#%% Cell 5: Final Visualizations & Rollout
# Plot Training Loss
plt.figure(figsize=(8, 4))
plt.plot(all_losses)
plt.yscale('log')
plt.title("Chunked Training Loss History")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.show()



#%%

# def plot_pred_ref_videos_rollout(video, ref_video, title="Rollout Render"):
#     """Specialized plotting function for the extended rollout sequence."""
#     fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    
#     t_mid = 23 # End of the original available data
#     t_end = len(video) - 1 # End of the forecast (t=47)

#     # t_mid = 3
#     # t_end = 6

#     sbimshow(video[0], title=f"{title} t=0", ax=ax[0, 0])
#     sbimshow(video[t_mid], title=f"{title} t={t_mid} (End Ref)", ax=ax[0, 1])
#     sbimshow(video[t_end], title=f"{title} t={t_end} (Forecast)", ax=ax[0, 2])

#     sbimshow(ref_video[0], title="Ref t=0", ax=ax[1, 0])
#     sbimshow(ref_video[t_mid], title=f"Ref t={t_mid}", ax=ax[1, 1])
#     sbimshow(ref_video[t_end], title=f"Ref t={t_end} (Zeroes)", ax=ax[1, 2])
#     plt.tight_layout()
#     plt.show()

def plot_pred_ref_videos_rollout(video, ref_video, title="Rollout Render"):
    """Specialized plotting function for the extended rollout sequence."""
    """ A finer version with len(video) // 2 = 12 to visualise better. """
    t_plots = np.arange(0, len(video), 2)
    fig, ax = plt.subplots(2, len(t_plots), figsize=(4*len(t_plots), 8))

    for i, t in enumerate(t_plots[:]):
        # if i >= 3: break  # Only plot the first 3 frames for clarity
        sbimshow(video[t], title=f"{title} t={t}", ax=ax[0, i])
        sbimshow(ref_video[t], title=f"Ref t={t}", ax=ax[1, i])
    plt.tight_layout()
    plt.show()


# --- Extended 48-Frame Rollout ---
print(f"\nRolling out {ROLLOUT_STEPS} frames starting from actual t=0...")

# Pad the original video with zeroes to match our desired rollout length
padding_steps = ROLLOUT_STEPS - nb_frames
ref_video_long = jnp.concatenate((ref_video, jnp.zeros((padding_steps, H, W, C))), axis=0)

# Run Autoregressive Generation
final_video = eqx.filter_jit(model.get_thetas_and_preds)(ref_video_long, 0.0, key, coords_grid)

# Plot rollout
plot_pred_ref_videos_rollout(final_video, ref_video_long, title="")

# Plot Matrix A Before/After
A_final = model.A
subsample_step = 10 
vmin, vmax = -1e-4, 1e-4

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
im1 = axes[0].imshow(A_init[::subsample_step, ::subsample_step], cmap='viridis', vmin=vmin, vmax=vmax)
axes[0].set_title(f"Recurrence Matrix A (Init)\nSubsampled step={subsample_step}")
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(A_final[::subsample_step, ::subsample_step], cmap='viridis', vmin=vmin, vmax=vmax)
axes[1].set_title(f"Recurrence Matrix A (Final)\nSubsampled step={subsample_step}")
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()
# %%

start_idx = 14
final_video = eqx.filter_jit(model.get_thetas_and_preds)(ref_video_long[start_idx:], 0.0, key, coords_grid)
plot_pred_ref_videos_rollout(final_video, ref_video_long[start_idx:], title="")

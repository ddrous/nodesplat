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

NB_EPOCHS = 5
NB_ITER_PER_EPOCH = 300
PRINT_EVERY = 10
LEARNING_RATE = 1e-4
# NB_LOSS_FRAMES = 2
NB_LOSS_PIXELS = 50
P_FORCING = 0.5

ROOT_WIDTH = 32
ROOT_DEPTH = 3
NUM_FOURIER_FREQS = 12

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
    # t_mid = 1
    # t_end = 2
    
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

    # return x

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

class WARP(eqx.Module):
    A: jax.Array
    B: jax.Array
    theta_0: jax.Array

    
    root_structure: RootMLP = eqx.field(static=True)
    unravel_fn: callable = eqx.field(static=True)
    d_theta: int = eqx.field(static=True)
    num_freqs: int = eqx.field(static=True)
    frame_shape: tuple = eqx.field(static=True)

    def __init__(self, root_width, root_depth, num_freqs, frame_shape, key):
        k_root, k_A, k_B, k_t0 = jax.random.split(key, 4)
        self.num_freqs = num_freqs
        self.frame_shape = frame_shape
        
        # Spatial input only (no tau)
        coord_dim = 2 + 2 * 2 * num_freqs 
        template_root = RootMLP(coord_dim, 7, root_width, root_depth, k_root)

        # template_root = RootMLP(coord_dim, 3, root_width, root_depth, k_root)
        # template_root = RootMLP(2, 3, root_width, root_depth, k_root)

        flat_params, self.unravel_fn = ravel_pytree(template_root)
        self.d_theta = flat_params.shape[0]
        self.root_structure = template_root
        
        # B maps from flattened image difference to weight space
        flat_image_size = frame_shape[0] * frame_shape[1] * frame_shape[2]
        
        self.A = jnp.eye(self.d_theta)
        # self.B = jax.random.normal(k_B, (self.d_theta, flat_image_size)) * 1e-4
        self.B = jnp.zeros((self.d_theta, flat_image_size))
        # self.theta_0 = jax.random.normal(k_t0, (self.d_theta,)) * 0.01
        self.theta_0 = flat_params

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
            
            # return rgb_fg

        return jax.vmap(render_pt)(thetas, coords)

    def get_thetas_and_preds(self, ref_video, p_forcing, key, coords_grid):
        H, W, C = self.frame_shape
        flat_coords = coords_grid.reshape(-1, 2)
        
        def scan_step(state, gt_curr_frame):
            theta, prev_frame_selected, k = state
            k, subk = jax.random.split(k)
            
            # 1. Render current frame
            thetas_frame = jnp.tile(theta, (H*W, 1))
            pred_flat = self.render_pixels(thetas_frame, flat_coords)
            pred_frame = pred_flat.reshape(H, W, C)
            
            # 2. Bernoulli Teacher Forcing decision
            use_gt = jax.random.bernoulli(subk, p_forcing)
            frame_t = jnp.where(use_gt, gt_curr_frame, pred_frame)
            
            # 3. Compute frame difference
            dx = frame_t - prev_frame_selected
            # dx_flat = dx.reshape(-1)
            dx_flat = dx.reshape(-1) / jnp.sqrt(H * W * C)
            
            # 4. Weight-space Recurrence Update
            theta_next = self.A @ theta + self.B @ dx_flat
            
            new_state = (theta_next, frame_t, subk)
            # new_state = (theta_next, gt_curr_frame, subk)

            return new_state, pred_frame
            
        init_frame = jnp.zeros((H, W, C))
        init_state = (self.theta_0, init_frame, key)
        
        # Scan over the sequence length
        _, pred_video = jax.lax.scan(scan_step, init_state, ref_video)
        return pred_video


#%% Cell 4: Initialization & Training
key, subkey = jax.random.split(key)
model = WARP(ROOT_WIDTH, ROOT_DEPTH, NUM_FOURIER_FREQS, (H, W, C), subkey)
A_init = model.A.copy()

scheduler = optax.exponential_decay(LEARNING_RATE, transition_steps=NB_ITER_PER_EPOCH*NB_EPOCHS, decay_rate=0.1)
optimizer = optax.adam(scheduler)
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

@eqx.filter_jit
def train_step(model, opt_state, key, ref_video, coords_grid, p_forcing):
    def loss_fn(m):
        pred_video = m.get_thetas_and_preds(ref_video, p_forcing, key, coords_grid)
        return jnp.mean(jnp.abs(pred_video - ref_video))  # L1 MAE Loss on full video

        # ## Randlomly select nb_loss_frames frames for loss computation
        # frame_indices = jax.random.choice(key, pred_video.shape[0], (NB_LOSS_FRAMES,), replace=False)
        # pred_selected = pred_video[frame_indices]
        # ref_selected = ref_video[frame_indices]
        # return jnp.mean((pred_selected - ref_selected)**2)

        # ## Randomly select pixels for loss computation
        # h_indices = jax.random.choice(key, pred_video.shape[1], (NB_LOSS_PIXELS,), replace=True)
        # w_indices = jax.random.choice(key, pred_video.shape[2], (NB_LOSS_PIXELS,), replace=True)
        # pred_selected = pred_video[:, h_indices, w_indices]
        # ref_selected = ref_video[:, h_indices, w_indices]
        # return jnp.mean((pred_selected - ref_selected)**2)

    # def loss_fn(m):
    #         pred_video = m.get_thetas_and_preds(ref_video, p_forcing, key, coords_grid)
            
    #         # --- CRITICAL FIX: Use the full video loss ---
    #         loss_full = jnp.mean(jnp.abs(pred_video - ref_video))
            
    #         # Add a 2x loss boost explicitly for frame 0 to force theta_0 to wake up quickly
    #         loss_t0 = jnp.mean(jnp.abs(pred_video[0] - ref_video[0]))
            
    #         return loss_full + 100.0 * loss_t0

    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val

# Initial Render Plot (Autoregressive, P=0)
print("Rendering Initial Video...")
init_video = model.get_thetas_and_preds(ref_video, p_forcing=0.0, key=key, coords_grid=coords_grid)
plot_pred_ref_videos(init_video, ref_video, title="Init (P_forcing=0)")

all_losses = []
start_time = time.time()

for epoch in range(NB_EPOCHS):
    print(f"\nEPOCH: {epoch+1}")
    epoch_losses = []
    
    pbar = tqdm(range(NB_ITER_PER_EPOCH))
    for i in pbar:
        key, subkey = jax.random.split(key)
        model, opt_state, loss = train_step(model, opt_state, subkey, ref_video, coords_grid, P_FORCING)
        epoch_losses.append(loss)
        
        if i % PRINT_EVERY == 0:
            pbar.set_description(f"Loss: {loss:.4f}")
            
    all_losses.extend(epoch_losses)
    
    # Render progress at end of epoch using P_FORCING = ?
    current_video = eqx.filter_jit(model.get_thetas_and_preds)(ref_video, P_FORCING, key, coords_grid)
    plot_pred_video(current_video, title=f"Epoch {epoch+1} Render")

wall_time = time.time() - start_time
print("\nWall time for WARP training in h:m:s:", time.strftime("%H:%M:%S", time.gmtime(wall_time)))

#%% Cell 5: Final Visualizations
# Final Video (Autoregressive, P=0)
final_video = eqx.filter_jit(model.get_thetas_and_preds)(ref_video, 0.0, key, coords_grid)
plot_pred_ref_videos(final_video, ref_video, title="Final (P_forcing=0)")

# Plot Loss
plt.figure(figsize=(8, 4))
plt.plot(all_losses)
plt.yscale('log')
plt.title("Loss History")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Plot Matrix A Before/After
A_final = model.A
subsample_step = max(1, model.d_theta // 100) 

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
im1 = axes[0].imshow(A_init[::subsample_step, ::subsample_step], cmap='viridis')
axes[0].set_title(f"Recurrence Matrix A (Init)\nSubsampled step={subsample_step}")
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(A_final[::subsample_step, ::subsample_step], cmap='viridis')
axes[1].set_title(f"Recurrence Matrix A (Final)\nSubsampled step={subsample_step}")
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()
import jax
import jax.numpy as jnp
from src.simulation.jax.composer import jax_superblock, jax_hermite_phi_matrix

# Configuration
cutoff = 10
hom_x = 0.0

# Create Inputs
# Leaf 0: Active, Squeezed (r=0.5)
# Leaf 1: Inactive, Squeezed (r=0.5) but Inactive flag = False
# If logic works, Mixer 0 (Inputs 0, 1) should IGNORE Leaf 1 and Pass Leaf 0.
# The result should be pure squeezed state.
# If logic fails (Old behavior), it will Mix(0, 1), adding vacuum noise -> Mixed state.

# Setup Leaves
# We need 8 leaves for superblock
n_leaves = 8
# Params... we can cheat and pass pre-computed states to jax_superblock?
# No, jax_superblock takes leaf_states.
# Let's mock the inputs to jax_superblock's mix logic directly?
# No, easier to run jax_superblock with dummy inputs.

# 1. Create Squeezed State (r=0.5)
import qutip as qt

state_sq = qt.squeeze(cutoff, 0.5) * qt.basis(cutoff, 0)
vec_sq = jnp.array(state_sq.full().flatten())
vec_vac = jnp.array(qt.basis(cutoff, 0).full().flatten())

# Leaves array
leaf_states = jnp.array([vec_sq if i == 0 else vec_vac for i in range(8)])
leaf_probs = jnp.ones(8)
leaf_active = jnp.array([True, False] + [False] * 6)  # Leaf 0 Active, Leaf 1 Inactive
leaf_pnr = jnp.zeros(8)
leaf_total_pnr = jnp.zeros(8)
leaf_modes = jnp.ones(8)

# Mix Params: Node 0 set to MIX (theta=pi/4)
mix_params = jnp.zeros((7, 3))
mix_params = mix_params.at[0, 0].set(jnp.pi / 4)  # 50/50 BS

# Context
phi_vec = jax_hermite_phi_matrix(jnp.array([0.0]), cutoff)[:, 0]
V_matrix = jnp.zeros((cutoff, 1))
dx_weights = jnp.zeros(1)

print("Running jax_superblock with Active/Inactive Fix...")
result = jax_superblock(
    leaf_states,
    leaf_probs,
    leaf_active,
    leaf_pnr,
    leaf_total_pnr,
    leaf_modes,
    mix_params,
    hom_x,
    0.0,
    0.0,
    phi_vec,
    V_matrix,
    dx_weights,
    cutoff,
    True,
    False,
    True,  # hom_window_none, hom_x_none -> False, hom_res_none
)

final_state = result[0]
purity = jnp.abs(jnp.trace(final_state @ final_state)) if final_state.ndim == 2 else 1.0

print(f"Final State Shape: {final_state.shape}")
print(f"Purity: {purity:.4f}")

# Expected:
# If Fixed: Purity = 1.0 (Pure state passed through)
# If Broken: Purity < 1.0 (Mixed with vacuum)
if purity > 0.99:
    print("SUCCESS: Logic correctly ignored inactive leaf.")
else:
    print("FAILURE: Logic mixed vacuum! Purity lost.")

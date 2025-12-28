import jax.numpy as jnp
from typing import Dict, Any
from abc import ABC, abstractmethod

# Defaults (can be overridden via config)
DEFAULT_CONFIG = {
    "pnr_max": 3,
    "window": 0.1,
    "hx_scale": 4.0,
    "r_scale": 2.0,
    "d_scale": 3.0,
}


class BaseGenotype(ABC):
    """Abstract base class for Genotype decoding strategies."""

    def __init__(self, depth: int = 3, config: Dict[str, Any] = None):
        self.depth = depth
        self.config = config if config is not None else DEFAULT_CONFIG

        self.r_scale = self.config.get("r_scale", 2.0)
        self.d_scale = self.config.get("d_scale", 3.0)
        self.hx_scale = self.config.get("hx_scale", 4.0)
        self.window = self.config.get("window", 0.1)
        self.window = self.config.get("window", 0.1)
        self.pnr_max = int(self.config.get("pnr_max", 3))
        self.active_threshold = 0.0

        # Dynamic Modes (1 Signal + N-1 Controls)
        self.n_modes = int(self.config.get("modes", 2))
        self.n_control = max(1, self.n_modes - 1)

    @abstractmethod
    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        """
        Decodes a genotype vector into a dictionary of circuit parameters.
        Must return a dict with keys:
          - homodyne_x: float
          - homodyne_window: float
          - mix_params: (7, 3) array [theta, phi, varphi]
          - mix_source: (7,) int array
          - leaf_active: (8,) bool array
          - leaf_params: dict containing leaf-specific arrays
          - final_gauss: dict {r, phi, varphi, disp}
        """
        pass

    @abstractmethod
    def get_length(self, depth: int = 3) -> int:
        """Returns the expected length of the genotype vector."""
        pass

    def _pad_genotype(self, g: jnp.ndarray) -> jnp.ndarray:
        return g


class LegacyGenotype(BaseGenotype):
    """
    Original "Legacy" Genotype implementation tailored to canonical output.
    Maintains 256 length.
    Maps old parameters to new structure:
      - tmss_r: Takes first column (index 0) of old 2-column params.
      - final_gauss: Returns identity (zeros).
    """

    def get_length(self, depth: int = 3) -> int:
        return 256

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        idx = 0

        # 1. Homodyne
        hom_x_raw = g[idx]
        idx += 1
        hom_win_raw = g[idx]
        idx += 1

        homodyne_x = jnp.tanh(hom_x_raw) * self.hx_scale
        homodyne_window = jnp.abs(
            jnp.tanh(hom_win_raw) * 20.0 * self.window
        )  # Legacy factor? Or just assume window usage?
        # WAIT: The original code had H_WINDOW = 0.1, but legacy decode used specific logic:
        # homodyne_window = jnp.abs(jnp.tanh(hom_win_raw) * 2.0)
        # So legacy ignored H_WINDOW constant?
        # Checking lines 63-64 of original file...
        # homodyne_window = jnp.abs(jnp.tanh(hom_win_raw) * 2.0)
        # So for Legacy, we should probably keep 2.0 or make it configurable?
        # User said "Use the current values as defaults".
        # Current Legacy implementation uses *2.0 hardcoded* (line 64).
        # H_WINDOW (0.1) was used in Design A/B/C.
        # Let's keep Legacy behavior as is (hardcoded 2.0) or map validly.
        # If user sets --window, should Legacy change? Probably yes.
        # If default window is 0.1, 2.0 is 20x.
        # Let's stick to using self.window logic if possible, or leave legacy alone if "Legacy" implies fixed behavior.
        # I will leave Legacy using its own hardcoded 2.0 unless user explicitly requested changing Legacy.
        # User said "All of these limits".
        # But Legacy uses *variable* window, while others use *fixed* H_WINDOW.
        # So for Legacy, `window` config might define the *scale*?
        # Let's leave Legacy as-is (hardcoded 2.0) but use self.hx_scale.

        homodyne_x = jnp.tanh(hom_x_raw) * self.hx_scale
        homodyne_window = jnp.abs(jnp.tanh(hom_win_raw) * 2.0)

        # Legacy rounding
        homodyne_x = jnp.round(homodyne_x * 1e6) / 1e6
        homodyne_window = jnp.round(homodyne_window * 1e6) / 1e6

        # 2. Mix Nodes (7)
        n_mix = 7
        mix_params_flat = g[idx : idx + n_mix * 4]
        idx += n_mix * 4
        mix_params_reshaped = mix_params_flat.reshape((n_mix, 4))

        mix_angles = jnp.tanh(mix_params_reshaped[:, :3]) * (jnp.pi / 2)
        # Source removed
        mix_source = jnp.zeros(n_mix, dtype=jnp.int32)

        # 3. Leaves (8)
        n_leaves = 8
        n_leaf_params = 17
        leaves_flat = g[idx : idx + n_leaves * n_leaf_params]
        leaves_reshaped = leaves_flat.reshape((n_leaves, n_leaf_params))

        # Param 0: Active
        leaf_active = leaves_reshaped[:, 0] > 0.0

        # Param 1: Num Controls
        n_ctrl_raw = leaves_reshaped[:, 1]
        leaf_n_ctrl = jnp.ones(n_leaves, dtype=jnp.int32)
        leaf_n_ctrl = jnp.where(n_ctrl_raw < -0.33, 0, leaf_n_ctrl)
        leaf_n_ctrl = jnp.where(n_ctrl_raw > 0.33, 2, leaf_n_ctrl)

        # Params 2-3: TMSS -> Take only column 0 for canonical form
        # Shape becomes (L,) instead of (L, 2)
        tmss_r_full = jnp.tanh(leaves_reshaped[:, 2:4]) * self.r_scale
        tmss_r = tmss_r_full[:, 0]

        # Param 4: US Phase (L,1)
        us_phase = jnp.tanh(leaves_reshaped[:, 4:5]) * (jnp.pi / 2)

        # Params 5-8: UC (4) -> (L, 4)
        # We split them into sub-arrays if needed or keep as block
        uc_params = leaves_reshaped[:, 5:9]
        uc_theta = jnp.tanh(uc_params[:, 0:1]) * (jnp.pi / 2)
        uc_phi = jnp.tanh(uc_params[:, 1:2]) * (jnp.pi / 2)
        uc_varphi = jnp.tanh(uc_params[:, 2:4]) * (jnp.pi / 2)

        # Params 9-10: Disp S
        disp_s_params = leaves_reshaped[:, 9:11]
        disp_s = (
            jnp.tanh(disp_s_params[:, 0]) * self.d_scale
            + 1j * jnp.tanh(disp_s_params[:, 1]) * self.d_scale
        )
        disp_s = disp_s[:, None]

        # Params 11-14: Disp C
        disp_c_params = leaves_reshaped[:, 11:15]
        disp_c = (
            jnp.tanh(disp_c_params[:, 0::2]) * self.d_scale
            + 1j * jnp.tanh(disp_c_params[:, 1::2]) * self.d_scale
        )

        # Params 15-16: PNR
        pnr_raw = jnp.clip(leaves_reshaped[:, 15:17], 0.0, 1.0)
        pnr = jnp.round(pnr_raw * self.pnr_max).astype(jnp.int32)

        leaf_params = {
            "n_ctrl": leaf_n_ctrl,
            "tmss_r": tmss_r,
            "us_phase": us_phase,
            "uc_theta": uc_theta,
            "uc_phi": uc_phi,
            "uc_varphi": uc_varphi,
            "disp_s": disp_s,
            "disp_c": disp_c,
            "pnr": pnr,
            "pnr_max": jnp.full((n_leaves,), self.pnr_max, dtype=jnp.int32),
        }

        # Final Gaussian: Identity (all zeros)
        final_gauss = {"r": 0.0, "phi": 0.0, "varphi": 0.0, "disp": 0.0 + 0.0j}

        return {
            "homodyne_x": homodyne_x,
            "homodyne_window": homodyne_window,
            "mix_params": mix_angles,
            "mix_source": mix_source,
            "leaf_active": leaf_active,
            "leaf_params": leaf_params,
            "final_gauss": final_gauss,
        }


class DesignAGenotype(BaseGenotype):
    """
    Design A: Original (per-leaf unique).
    Updated for Canonical Form:
      - 1 TMSS per leaf
      - Final Global Gaussian (5 params)
    """

    def __init__(self, depth: int = 3, config: Dict[str, Any] = None):
        super().__init__(depth, config)
        self.leaves = 2**depth
        self.nodes = self.leaves - 1

        # Calculate P_leaf_full dynamically
        n_c = self.n_control
        n_pairs = (n_c * (n_c - 1)) // 2
        len_uc = 2 * n_pairs + n_c
        len_disp_c = 2 * n_c
        len_pnr = n_c
        # Layout: Active(1), NCtrl(1), TMSS(1), US(1), UC, DispS(2), DispC, PNR
        self.P_leaf_full = 1 + 1 + 1 + 1 + len_uc + 2 + len_disp_c + len_pnr

        self.PN = 3
        self.G = 1
        self.F = 5

    def get_length(self, depth: int = 3) -> int:
        L = 2**depth

        # Per leaf:
        n_ctrl_modes = self.n_control

        # 0: Active (1)
        # 1: Num Ctrl (1)
        # 2: TMSS (1) -> Changed from 2 to 1 for Canonical
        # 3: US Phase (1)
        # 4...: UC + Disp + PNR

        n_uc_pairs = (n_ctrl_modes * (n_ctrl_modes - 1)) // 2
        len_uc = 2 * n_uc_pairs + n_ctrl_modes
        len_disp_c = 2 * n_ctrl_modes
        len_pnr = n_ctrl_modes

        # Total per leaf:
        params_per_leaf = 1 + 1 + 1 + 1 + len_uc + 2 + len_disp_c + len_pnr

        # Global
        # Hom(1)
        # Mix(Nodes * 3)
        len_mix = (L - 1) * 3
        # Final Gauss (5)
        len_final = 5

        return 1 + L * params_per_leaf + len_mix + len_final

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        idx = 0

        # 1. Global Homodyne X
        hom_x_raw = g[idx]
        idx += 1
        homodyne_x = jnp.tanh(hom_x_raw) * self.hx_scale
        homodyne_window = self.window

        # 2. Leaves (L * 16)
        n_leaves = self.leaves
        # Calculate n_leaf_params dynamically based on n_control
        n_ctrl_modes = self.n_control
        n_uc_pairs = (n_ctrl_modes * (n_ctrl_modes - 1)) // 2
        len_uc = 2 * n_uc_pairs + n_ctrl_modes
        len_disp_c = 2 * n_ctrl_modes
        len_pnr = n_ctrl_modes
        n_leaf_params = 1 + 1 + 1 + 1 + len_uc + 2 + len_disp_c + len_pnr

        leaves_flat = g[idx : idx + n_leaves * n_leaf_params]
        idx += n_leaves * n_leaf_params
        leaves_reshaped = leaves_flat.reshape((n_leaves, n_leaf_params))

        # Decode Leaf Block
        # 0: Active
        leaf_active = leaves_reshaped[:, 0] > 0.0

        # 1: Num Controls
        n_ctrl_raw = leaves_reshaped[:, 1]
        # Map [-1, 1] to [0, n_control]
        norm_val = (n_ctrl_raw + 1.0) / 2.0  # 0..1
        leaf_n_ctrl = jnp.round(norm_val * self.n_control).astype(jnp.int32)
        leaf_n_ctrl = jnp.clip(leaf_n_ctrl, 0, self.n_control)

        # 2: TMSS (Single Scalar)
        tmss_r = jnp.tanh(leaves_reshaped[:, 2]) * self.r_scale

        # 3: US Phase
        us_phase = jnp.tanh(leaves_reshaped[:, 3:4]) * (jnp.pi / 2)

        # 4...: UC Params
        # Start idx 4. Len calculated above.
        N_C = self.n_control
        n_uc_pairs = (N_C * (N_C - 1)) // 2
        # theta, phi per pair
        len_pairs = 2 * n_uc_pairs
        len_phases = N_C
        len_uc = len_pairs + len_phases

        uc_slice = leaves_reshaped[:, 4 : 4 + len_uc]
        idx_offset = 4 + len_uc

        # Split into theta, phi, varphi
        # For N=1 (Control=1), pairs=0. Just phase (1).
        # For N=2 (Control=2), pairs=1. Theta, Phi, Ph1, Ph2. Total 4.
        # For N=3 (Control=3), pairs=3. 3*2 + 3 = 9.

        if n_uc_pairs > 0:
            uc_pairs_raw = uc_slice[:, :len_pairs]
            uc_theta = jnp.tanh(uc_pairs_raw[:, 0::2]) * (jnp.pi / 2)
            uc_phi = jnp.tanh(uc_pairs_raw[:, 1::2]) * (jnp.pi / 2)
        else:
            uc_theta = jnp.zeros((n_leaves, 0))
            uc_phi = jnp.zeros((n_leaves, 0))

        uc_varphi_raw = uc_slice[:, len_pairs:]
        uc_varphi = jnp.tanh(uc_varphi_raw) * (jnp.pi / 2)

        # 6. Disp S (2 params)
        disp_s_slice = leaves_reshaped[:, idx_offset : idx_offset + 2]
        idx_offset += 2

        disp_s = (
            jnp.tanh(disp_s_slice[:, 0]) * self.d_scale
            + 1j * jnp.tanh(disp_s_slice[:, 1]) * self.d_scale
        )
        disp_s = disp_s[:, None]

        # 7. Disp C (2 * N_C params)
        len_disp_c = 2 * N_C
        disp_c_slice = leaves_reshaped[:, idx_offset : idx_offset + len_disp_c]
        idx_offset += len_disp_c

        disp_c = (
            jnp.tanh(disp_c_slice[:, 0::2]) * self.d_scale
            + 1j * jnp.tanh(disp_c_slice[:, 1::2]) * self.d_scale
        )

        # 8. PNR (N_C params)
        len_pnr = N_C
        pnr_slice = leaves_reshaped[:, idx_offset : idx_offset + len_pnr]

        pnr_raw = jnp.clip(pnr_slice, 0.0, 1.0)
        pnr = jnp.round(pnr_raw * self.pnr_max).astype(jnp.int32)

        leaf_params = {
            "n_ctrl": leaf_n_ctrl,
            "tmss_r": tmss_r,
            "us_phase": us_phase,
            "uc_theta": uc_theta,
            "uc_phi": uc_phi,
            "uc_varphi": uc_varphi,
            "disp_s": disp_s,
            "disp_c": disp_c,
            "pnr": pnr,
            "pnr_max": jnp.full((n_leaves,), self.pnr_max, dtype=jnp.int32),
        }

        # 3. Mix Nodes
        n_mix = self.nodes
        mix_params_flat = g[idx : idx + n_mix * self.PN]
        idx += n_mix * self.PN
        mix_reshaped = mix_params_flat.reshape((n_mix, self.PN))

        mix_angles = jnp.tanh(mix_reshaped[:, :3]) * (jnp.pi / 2)
        # Source removed
        mix_source = jnp.zeros(n_mix, dtype=jnp.int32)

        # 4. Final Gaussian (5)
        final_raw = g[idx : idx + self.F]
        idx += self.F
        r_final = jnp.tanh(final_raw[0]) * self.r_scale
        phi_final = jnp.tanh(final_raw[1]) * (jnp.pi / 2)
        varphi_final = jnp.tanh(final_raw[2]) * (jnp.pi / 2)
        disp_final = (
            jnp.tanh(final_raw[3]) * self.d_scale
            + 1j * jnp.tanh(final_raw[4]) * self.d_scale
        )

        final_gauss = {
            "r": r_final,
            "phi": phi_final,
            "varphi": varphi_final,
            "disp": disp_final,
        }

        return {
            "homodyne_x": homodyne_x,
            "homodyne_window": homodyne_window,
            "mix_params": mix_angles,
            "mix_source": mix_source,
            "leaf_active": leaf_active,
            "leaf_params": leaf_params,
            "final_gauss": final_gauss,
        }


class DesignB1Genotype(BaseGenotype):
    """
    Design B1: Tied-leaf, no per-leaf active.
    BP = 15 (was 16).
    """

    def __init__(self, depth: int = 3, config: Dict[str, Any] = None):
        super().__init__(depth, config)
        self.leaves = 2**depth
        self.nodes = self.leaves - 1
        # BP = 15, PN = 3, G = 1, F = 5
        self.BP = 15
        self.PN = 3
        self.G = 1
        self.F = 5

    def get_length(self, depth: int = 3) -> int:
        L = 2**depth

        # B1 Length:
        # 1(Hom) + BP(Shared) + 4*(L-1)(Mix) + 5(Final)
        L = 2**depth

        # Shared Block (BP)
        # Old: ~... + 2(TMSS) ...
        # New: ~... + 1(TMSS) ...
        # Calculate BP dynamically
        N_C = self.n_control
        n_uc_pairs = (N_C * (N_C - 1)) // 2
        len_uc = 2 * n_uc_pairs + N_C
        len_disp_c = 2 * N_C
        len_pnr = N_C

        # Shared Block Params:
        # 0: NCtrl(1)
        # 1: TMSS(1)
        # 2: US(1)
        # 3...: UC + Disp + PNR
        # BP = 1 + 1 + 1 + len_uc + 2 + len_disp_c + len_pnr
        self.BP = 1 + 1 + 1 + len_uc + 2 + len_disp_c + len_pnr

        # Mix = 3 * (L-1)

        return 1 + self.BP + 3 * (L - 1) + 5

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        idx = 0
        hom_x_raw = g[idx]
        idx += 1
        homodyne_x = jnp.tanh(hom_x_raw) * self.hx_scale
        homodyne_window = self.window

        # Shared Params
        N_C = self.n_control
        n_uc_pairs = (N_C * (N_C - 1)) // 2
        len_uc = 2 * n_uc_pairs + N_C
        len_disp_c = 2 * N_C
        len_pnr = N_C
        BP = 1 + 1 + 1 + len_uc + 2 + len_disp_c + len_pnr

        bp_raw = g[idx : idx + BP]
        idx += BP

        # 0: n_ctrl
        val = bp_raw[0]
        norm_val = (val + 1.0) / 2.0
        n_ctrl_int = jnp.round(norm_val * self.n_control).astype(jnp.int32)
        n_ctrl_int = jnp.clip(n_ctrl_int, 0, self.n_control)

        # 1: tmss_r (Single Scalar)
        tmss_r_val = jnp.tanh(bp_raw[1]) * self.r_scale

        # 2: us_phase
        us_phase_val = jnp.tanh(bp_raw[2:3]) * (jnp.pi / 2)

        # 3...: UC
        len_pairs = 2 * n_uc_pairs
        uc_slice = bp_raw[3 : 3 + len_uc]

        if n_uc_pairs > 0:
            uc_pairs_raw = uc_slice[:len_pairs]
            uc_theta_val = jnp.tanh(uc_pairs_raw[0::2]) * (jnp.pi / 2)
            uc_phi_val = jnp.tanh(uc_pairs_raw[1::2]) * (jnp.pi / 2)
        else:
            uc_theta_val = jnp.zeros(0)
            uc_phi_val = jnp.zeros(0)

        uc_varphi_raw = uc_slice[len_pairs:]
        uc_varphi_val = jnp.tanh(uc_varphi_raw) * (jnp.pi / 2)

        idx_offset = 3 + len_uc

        # Disp S
        disp_s_raw = bp_raw[idx_offset : idx_offset + 2]
        idx_offset += 2
        disp_s_val = (
            jnp.tanh(disp_s_raw[0]) * self.d_scale
            + 1j * jnp.tanh(disp_s_raw[1]) * self.d_scale
        )
        disp_s_val = disp_s_val[None]

        # Disp C
        disp_c_raw = bp_raw[idx_offset : idx_offset + len_disp_c]
        idx_offset += len_disp_c
        disp_c_val = (
            jnp.tanh(disp_c_raw[0::2]) * self.d_scale
            + 1j * jnp.tanh(disp_c_raw[1::2]) * self.d_scale
        )

        # PNR
        pnr_raw = jnp.clip(bp_raw[idx_offset : idx_offset + len_pnr], 0.0, 1.0)
        pnr_val = jnp.round(pnr_raw * self.pnr_max).astype(jnp.int32)

        # Broadcast
        n_leaves = self.leaves
        leaf_params = {
            "n_ctrl": jnp.broadcast_to(n_ctrl_int, (n_leaves,)),
            "tmss_r": jnp.broadcast_to(
                tmss_r_val, (n_leaves,)
            ),  # Single scalar per leaf (L,)
            "us_phase": jnp.broadcast_to(us_phase_val, (n_leaves, 1)),
            "uc_theta": jnp.broadcast_to(uc_theta_val, (n_leaves, n_uc_pairs)),
            "uc_phi": jnp.broadcast_to(uc_phi_val, (n_leaves, n_uc_pairs)),
            "uc_varphi": jnp.broadcast_to(uc_varphi_val, (n_leaves, N_C)),
            "disp_s": jnp.broadcast_to(disp_s_val, (n_leaves, 1)),
            "disp_c": jnp.broadcast_to(disp_c_val, (n_leaves, N_C)),
            "pnr": jnp.broadcast_to(pnr_val, (n_leaves, N_C)),
            "pnr_max": jnp.full((n_leaves,), self.pnr_max, dtype=jnp.int32),
        }
        leaf_active = jnp.ones(n_leaves, dtype=bool)

        # Mix Params
        n_mix = self.nodes
        mix_params_flat = g[idx : idx + n_mix * self.PN]
        idx += n_mix * self.PN
        mix_reshaped = mix_params_flat.reshape((n_mix, self.PN))
        mix_angles = jnp.tanh(mix_reshaped[:, :3]) * (jnp.pi / 2)
        # Source removed
        mix_source = jnp.zeros(n_mix, dtype=jnp.int32)

        # Final Gaussian
        final_raw = g[idx : idx + self.F]
        idx += self.F
        r_final = jnp.tanh(final_raw[0]) * self.r_scale
        phi_final = jnp.tanh(final_raw[1]) * (jnp.pi / 2)
        varphi_final = jnp.tanh(final_raw[2]) * (jnp.pi / 2)
        disp_final = (
            jnp.tanh(final_raw[3]) * self.d_scale
            + 1j * jnp.tanh(final_raw[4]) * self.d_scale
        )
        final_gauss = {
            "r": r_final,
            "phi": phi_final,
            "varphi": varphi_final,
            "disp": disp_final,
        }

        return {
            "homodyne_x": homodyne_x,
            "homodyne_window": homodyne_window,
            "mix_params": mix_angles,
            "mix_source": mix_source,
            "leaf_active": leaf_active,
            "leaf_params": leaf_params,
            "final_gauss": final_gauss,
        }


class DesignB3Genotype(BaseGenotype):
    """
    Design B3: Semi-Tied.
    - Shared: Continuous Block (TMSS, US, UC, DispS, DispC).
    - Unique Per Leaf: Discrete Block (Active, NCtrl, PNR).
    """

    def __init__(self, depth: int = 3, config: Dict[str, Any] = None):
        super().__init__(depth, config)
        self.leaves = 2**depth
        self.nodes = self.leaves - 1

        # Calculate lengths
        N_C = self.n_control
        n_uc_pairs = (N_C * (N_C - 1)) // 2

        # Shared Continuous Block length
        # TMSS(1) + US(1) + UC(len_uc) + DispS(2) + DispC(2*N_C)
        len_uc = 2 * n_uc_pairs + N_C
        len_disp_c = 2 * N_C
        self.Sharedv = 1 + 1 + len_uc + 2 + len_disp_c

        # Unique Discrete Block length (Per Leaf)
        # Active(1) + NCtrl(1) + PNR(N_C)
        self.Unique = 1 + 1 + N_C

        self.F = 5

    def get_length(self, depth: int = 3) -> int:
        L = 2**depth
        # Total = Hom(1) + Shared + L*Unique + Mix(3*(L-1)) + Final(5)
        return 1 + self.Sharedv + L * self.Unique + 3 * (L - 1) + 5

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        idx = 0

        # 1. Homodyne
        hom_x_raw = g[idx]
        idx += 1
        homodyne_x = jnp.tanh(hom_x_raw) * self.hx_scale
        homodyne_window = self.window

        # 2. Shared Continuous Block
        # Layout: TMSS, US, UC, DispS, DispC
        shared_raw = g[idx : idx + self.Sharedv]
        idx += self.Sharedv

        s_idx = 0
        # TMSS
        tmss_r = jnp.tanh(shared_raw[s_idx]) * self.r_scale
        s_idx += 1

        # US Phase
        us_phase = jnp.tanh(shared_raw[s_idx : s_idx + 1]) * (jnp.pi / 2)
        s_idx += 1

        # UC
        N_C = self.n_control
        n_uc_pairs = (N_C * (N_C - 1)) // 2
        len_uc = 2 * n_uc_pairs + N_C
        uc_raw = shared_raw[s_idx : s_idx + len_uc]
        s_idx += len_uc

        if n_uc_pairs > 0:
            uc_pairs = uc_raw[: 2 * n_uc_pairs]
            uc_theta = jnp.tanh(uc_pairs[0::2]) * (jnp.pi / 2)
            uc_phi = jnp.tanh(uc_pairs[1::2]) * (jnp.pi / 2)
        else:
            uc_theta = jnp.array([])
            uc_phi = jnp.array([])

        uc_varphi = jnp.tanh(uc_raw[2 * n_uc_pairs :]) * (jnp.pi / 2)

        # Disp S
        disp_s_sub = shared_raw[s_idx : s_idx + 2]
        s_idx += 2
        disp_s = (
            jnp.tanh(disp_s_sub[0]) * self.d_scale
            + 1j * jnp.tanh(disp_s_sub[1]) * self.d_scale
        )
        disp_s = disp_s[None]  # (1,)

        # Disp C
        len_disp_c = 2 * N_C
        disp_c_sub = shared_raw[s_idx : s_idx + len_disp_c]
        s_idx += len_disp_c
        disp_c = (
            jnp.tanh(disp_c_sub[0::2]) * self.d_scale
            + 1j * jnp.tanh(disp_c_sub[1::2]) * self.d_scale
        )

        # 3. Unique Discrete Block (Per Leaf)
        # Layout per leaf: Active, NCtrl, PNR
        # Shape (L, Unique)
        unique_total = self.leaves * self.Unique
        unique_raw = g[idx : idx + unique_total]
        idx += unique_total

        unique_reshaped = unique_raw.reshape((self.leaves, self.Unique))

        # Active (Idx 0)
        leaf_active = unique_reshaped[:, 0] > 0.0

        # NCtrl (Idx 1)
        n_ctrl_raw = unique_reshaped[:, 1]
        norm_val = (n_ctrl_raw + 1.0) / 2.0
        # Map roughly to 3 bins
        leaf_n_ctrl = jnp.round(norm_val * self.n_control).astype(jnp.int32)
        leaf_n_ctrl = jnp.clip(leaf_n_ctrl, 0, self.n_control)

        # PNR (Idx 2..2+N_C)
        pnr_raw = jnp.clip(unique_reshaped[:, 2 : 2 + N_C], 0.0, 1.0)
        leaf_pnr = jnp.round(pnr_raw * self.pnr_max).astype(jnp.int32)

        # Broadcast Shared Params to (L, ...)
        L = self.leaves

        leaf_params = {
            "n_ctrl": leaf_n_ctrl,  # (L,)
            "tmss_r": jnp.broadcast_to(tmss_r, (L,)),
            "us_phase": jnp.broadcast_to(us_phase, (L, 1)),
            "uc_theta": jnp.broadcast_to(uc_theta, (L, n_uc_pairs)),
            "uc_phi": jnp.broadcast_to(uc_phi, (L, n_uc_pairs)),
            "uc_varphi": jnp.broadcast_to(uc_varphi, (L, N_C)),
            "disp_s": jnp.broadcast_to(disp_s, (L, 1)),
            "disp_c": jnp.broadcast_to(disp_c, (L, N_C)),
            "pnr": leaf_pnr,  # (L, N_C)
            "pnr_max": jnp.full((L,), self.pnr_max, dtype=jnp.int32),
        }

        # 4. Mix Nodes and Final
        # Mix params reduced from 4 to 3
        mix_len = 3 * (self.nodes)
        if mix_len > 0:
            mix_raw = g[idx : idx + mix_len]
            idx += mix_len
            mix_reshaped = mix_raw.reshape((self.nodes, 3))
            mix_angles = jnp.tanh(mix_reshaped) * (jnp.pi / 2)
            # Source removed
            mix_source = jnp.zeros(self.nodes, dtype=jnp.int32)
        else:
            mix_angles = jnp.zeros((0, 3))
            mix_source = jnp.zeros((0,), dtype=jnp.int32)

        final_raw = g[idx : idx + 5]
        # Same as A/B
        r_final = jnp.tanh(final_raw[0]) * self.r_scale
        phi_final = jnp.tanh(final_raw[1]) * (jnp.pi / 2)
        varphi_final = jnp.tanh(final_raw[2]) * (jnp.pi / 2)
        disp_final = (
            jnp.tanh(final_raw[3]) * self.d_scale
            + 1j * jnp.tanh(final_raw[4]) * self.d_scale
        )
        final_gauss = {
            "r": r_final,
            "phi": phi_final,
            "varphi": varphi_final,
            "disp": disp_final,
        }

        return {
            "homodyne_x": homodyne_x,
            "homodyne_window": homodyne_window,
            "mix_params": mix_angles,
            "mix_source": mix_source,
            "leaf_active": leaf_active,
            "leaf_params": leaf_params,
            "final_gauss": final_gauss,
        }


class DesignB2Genotype(DesignB1Genotype):
    """
    Design B2: Tied-leaf with per-leaf active flags.
    Adds L booleans.
    """

    def get_length(self, depth: int = 3) -> int:
        L = 2**depth
        # B1 Length + L (Active flags)
        b1_len = super().get_length(depth)
        return b1_len + L

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        # B1 Length
        b1_len = super().get_length(self.depth)

        # Decode B1 part
        b1_part = g[:b1_len]
        decoded = super().decode(b1_part, cutoff)

        # Add Active Flags (Last self.leaves params)
        active_raw = g[b1_len : b1_len + self.leaves]
        leaf_active = active_raw > self.active_threshold
        decoded["leaf_active"] = leaf_active

        return decoded


class DesignC1Genotype(BaseGenotype):
    """
    Design C1: Tied-all, no active flags.
    """

    def __init__(self, depth: int = 3, config: Dict[str, Any] = None):
        super().__init__(depth, config)
        self.leaves = 2**depth
        self.nodes = self.leaves - 1
        self.BP = 15
        self.PN = 3
        self.G = 1
        self.F = 5

    def get_length(self, depth: int = 3) -> int:
        # C1 Length:
        # 1(Hom) + P_Shared + 4(Mix) + 5(Final)
        N_C = self.n_control
        n_uc_pairs = (N_C * (N_C - 1)) // 2
        len_uc = 2 * n_uc_pairs + N_C
        len_disp_c = 2 * N_C
        len_pnr = N_C

        params_shared = 1 + 1 + 1 + len_uc + 2 + len_disp_c + len_pnr

        return 1 + params_shared + 3 + 5

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        idx = 0
        hom_x_raw = g[idx]
        idx += 1
        homodyne_x = jnp.tanh(hom_x_raw) * self.hx_scale
        homodyne_window = self.window

        # Shared Block
        N_C = self.n_control
        n_uc_pairs = (N_C * (N_C - 1)) // 2
        len_uc = 2 * n_uc_pairs + N_C
        len_disp_c = 2 * N_C
        len_pnr = N_C
        P_shared = 1 + 1 + 1 + len_uc + 2 + len_disp_c + len_pnr

        bp_raw = g[idx : idx + P_shared]
        idx += P_shared

        # 0: n_ctrl
        val = bp_raw[0]
        norm_val = (val + 1.0) / 2.0
        n_ctrl_int = jnp.round(norm_val * self.n_control).astype(jnp.int32)
        n_ctrl_int = jnp.clip(n_ctrl_int, 0, self.n_control)

        # 1: tmss_r
        tmss_r_val = jnp.tanh(bp_raw[1]) * self.r_scale

        # 2: us_phase
        us_phase_val = jnp.tanh(bp_raw[2:3]) * (jnp.pi / 2)

        # 3...: UC
        len_pairs = 2 * n_uc_pairs
        uc_slice = bp_raw[3 : 3 + len_uc]

        if n_uc_pairs > 0:
            uc_pairs_raw = uc_slice[:len_pairs]
            uc_theta_val = jnp.tanh(uc_pairs_raw[0::2]) * (jnp.pi / 2)
            uc_phi_val = jnp.tanh(uc_pairs_raw[1::2]) * (jnp.pi / 2)
        else:
            uc_theta_val = jnp.zeros(0)
            uc_phi_val = jnp.zeros(0)

        uc_varphi_raw = uc_slice[len_pairs:]
        uc_varphi_val = jnp.tanh(uc_varphi_raw) * (jnp.pi / 2)

        idx_offset = 3 + len_uc

        # Disp S
        disp_s_raw = bp_raw[idx_offset : idx_offset + 2]
        idx_offset += 2
        disp_s_val = (
            jnp.tanh(disp_s_raw[0]) * self.d_scale
            + 1j * jnp.tanh(disp_s_raw[1]) * self.d_scale
        )
        disp_s_val = disp_s_val[None]

        # Disp C
        disp_c_raw = bp_raw[idx_offset : idx_offset + len_disp_c]
        idx_offset += len_disp_c
        disp_c_val = (
            jnp.tanh(disp_c_raw[0::2]) * self.d_scale
            + 1j * jnp.tanh(disp_c_raw[1::2]) * self.d_scale
        )

        # PNR
        pnr_raw = jnp.clip(bp_raw[idx_offset : idx_offset + len_pnr], 0.0, 1.0)
        pnr_val = jnp.round(pnr_raw * self.pnr_max).astype(jnp.int32)

        # Broadcast
        n_leaves = self.leaves
        leaf_params = {
            "n_ctrl": jnp.broadcast_to(n_ctrl_int, (n_leaves,)),
            "tmss_r": jnp.broadcast_to(tmss_r_val, (n_leaves,)),
            "us_phase": jnp.broadcast_to(us_phase_val, (n_leaves, 1)),
            "uc_theta": jnp.broadcast_to(uc_theta_val, (n_leaves, n_uc_pairs)),
            "uc_phi": jnp.broadcast_to(uc_phi_val, (n_leaves, n_uc_pairs)),
            "uc_varphi": jnp.broadcast_to(uc_varphi_val, (n_leaves, N_C)),
            "disp_s": jnp.broadcast_to(disp_s_val, (n_leaves, 1)),
            "disp_c": jnp.broadcast_to(disp_c_val, (n_leaves, N_C)),
            "pnr": jnp.broadcast_to(pnr_val, (n_leaves, N_C)),
            "pnr_max": jnp.full((n_leaves,), self.pnr_max, dtype=jnp.int32),
        }
        leaf_active = jnp.ones(n_leaves, dtype=bool)

        # Mix (Tied: 1 node)
        mix_params_slice = g[idx : idx + self.PN]
        idx += self.PN
        mix_params_reshaped = mix_params_slice.reshape((1, self.PN))
        mix_angles_one = jnp.tanh(mix_params_reshaped[:, :3]) * (jnp.pi / 2)

        # Source removed
        mix_src_one = jnp.zeros(1, dtype=jnp.int32)

        # Broadcast Mix
        n_mix = self.nodes
        mix_angles = jnp.broadcast_to(mix_angles_one, (n_mix, 3))
        mix_source = jnp.broadcast_to(mix_src_one, (n_mix,))

        # Final Gaussian
        final_raw = g[idx : idx + self.F]
        idx += self.F
        r_final = jnp.tanh(final_raw[0]) * self.r_scale
        phi_final = jnp.tanh(final_raw[1]) * (jnp.pi / 2)
        varphi_final = jnp.tanh(final_raw[2]) * (jnp.pi / 2)
        disp_final = (
            jnp.tanh(final_raw[3]) * self.d_scale
            + 1j * jnp.tanh(final_raw[4]) * self.d_scale
        )
        final_gauss = {
            "r": r_final,
            "phi": phi_final,
            "varphi": varphi_final,
            "disp": disp_final,
        }

        return {
            "homodyne_x": homodyne_x,
            "homodyne_window": homodyne_window,
            "mix_params": mix_angles,
            "mix_source": mix_source,
            "leaf_active": leaf_active,
            "leaf_params": leaf_params,
            "final_gauss": final_gauss,
        }


class DesignC2Genotype(DesignC1Genotype):
    """
    Design C2: Tied-all with per-leaf active.
    Length = 25 + L.
    """

    def get_length(self, depth: int = 3) -> int:
        c1_len = super().get_length(depth)
        L = 2**depth
        return c1_len + L

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        c1_len = super().get_length(self.depth)

        c1_part = g[:c1_len]
        decoded = super().decode(c1_part, cutoff)

        active_raw = g[c1_len : c1_len + self.leaves]
        leaf_active = active_raw > self.active_threshold
        decoded["leaf_active"] = leaf_active

        return decoded


class Design0Genotype(DesignAGenotype):
    """
    Design 0: Identical to Design A, but with per-node Homodyne Detection.
    Homodyne x is optimized for EACH mixing node (L-1 nodes).
    Range: [-5, 5] (controlled by hx_scale).
    """

    def __init__(self, depth: int = 3, config: Dict[str, Any] = None):
        super().__init__(depth, config)
        # Ensure hx_scale defaults to 5.0 if not specified, as requested (-5 to 5)
        if config is None or "hx_scale" not in config:
            self.hx_scale = 5.0
        # If config exists but key missing? Base class sets self.config.
        # But BaseClass __init__ runs first.
        # If user passed config without hx_scale, BaseClass used default 4.0.
        # We want 5.0.
        if config is None or "hx_scale" not in config:
            self.hx_scale = 5.0

    def get_length(self, depth: int = 3) -> int:
        # Base length for A
        # A has 1 global hom_x.
        # We need (L-1) hom_x.
        # So we add (L-1) - 1 = L-2 extra parameters?
        # A Length = 1 + ...
        # 0 Length = (L-1) + ...
        # Difference is L-2.

        base_len = super().get_length(depth)
        L = 2**depth
        return base_len + (L - 1) - 1

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        idx = 0

        # 1. Per-Node Homodyne X
        # Replaces global hom_x
        n_nodes = self.nodes  # L-1

        hom_x_raw = g[idx : idx + n_nodes]
        idx += n_nodes

        homodyne_x = jnp.tanh(hom_x_raw) * self.hx_scale
        homodyne_window = self.window

        # 2. Leaves (Same as A)
        # We can reuse A's decode logic for leaves/mix/final, but we need to offset the index?
        # A's decode starts at idx=0 reading 1 hom_x.
        # We read n_nodes hom_x.
        # We can pass the rest of g to a partially modified decode?
        # Or Just copy paste A's decode logic?
        # A's decode is monolithic. Copying is safer than super() hacks on slicing.

        # --- COPIED FROM DesignAGenotype (Starting after Homodyne) ---
        n_leaves = self.leaves
        n_ctrl_modes = self.n_control
        n_uc_pairs = (n_ctrl_modes * (n_ctrl_modes - 1)) // 2
        len_uc = 2 * n_uc_pairs + n_ctrl_modes
        len_disp_c = 2 * n_ctrl_modes
        len_pnr = n_ctrl_modes
        n_leaf_params = 1 + 1 + 1 + 1 + len_uc + 2 + len_disp_c + len_pnr

        leaves_flat = g[idx : idx + n_leaves * n_leaf_params]
        idx += n_leaves * n_leaf_params
        leaves_reshaped = leaves_flat.reshape((n_leaves, n_leaf_params))

        # Decode Leaf Block
        # 0: Active
        leaf_active = leaves_reshaped[:, 0] > 0.0

        # 1: Num Controls
        n_ctrl_raw = leaves_reshaped[:, 1]
        norm_val = (n_ctrl_raw + 1.0) / 2.0
        leaf_n_ctrl = jnp.round(norm_val * self.n_control).astype(jnp.int32)
        leaf_n_ctrl = jnp.clip(leaf_n_ctrl, 0, self.n_control)

        # 2: TMSS
        tmss_r = jnp.tanh(leaves_reshaped[:, 2]) * self.r_scale

        # 3: US Phase
        us_phase = jnp.tanh(leaves_reshaped[:, 3:4]) * (jnp.pi / 2)

        # 4...: UC Params
        len_pairs = 2 * n_uc_pairs
        len_uc = len_pairs + n_ctrl_modes
        uc_slice = leaves_reshaped[:, 4 : 4 + len_uc]
        idx_offset = 4 + len_uc

        if n_uc_pairs > 0:
            uc_pairs_raw = uc_slice[:, :len_pairs]
            uc_theta = jnp.tanh(uc_pairs_raw[:, 0::2]) * (jnp.pi / 2)
            uc_phi = jnp.tanh(uc_pairs_raw[:, 1::2]) * (jnp.pi / 2)
        else:
            uc_theta = jnp.zeros((n_leaves, 0))
            uc_phi = jnp.zeros((n_leaves, 0))

        uc_varphi_raw = uc_slice[:, len_pairs:]
        uc_varphi = jnp.tanh(uc_varphi_raw) * (jnp.pi / 2)

        # 6. Disp S
        disp_s_slice = leaves_reshaped[:, idx_offset : idx_offset + 2]
        idx_offset += 2
        disp_s = (
            jnp.tanh(disp_s_slice[:, 0]) * self.d_scale
            + 1j * jnp.tanh(disp_s_slice[:, 1]) * self.d_scale
        )
        disp_s = disp_s[:, None]

        # 7. Disp C
        len_disp_c = 2 * n_ctrl_modes
        disp_c_slice = leaves_reshaped[:, idx_offset : idx_offset + len_disp_c]
        idx_offset += len_disp_c
        disp_c = (
            jnp.tanh(disp_c_slice[:, 0::2]) * self.d_scale
            + 1j * jnp.tanh(disp_c_slice[:, 1::2]) * self.d_scale
        )

        # 8. PNR
        len_pnr = n_ctrl_modes
        pnr_slice = leaves_reshaped[:, idx_offset : idx_offset + len_pnr]
        pnr_raw = jnp.clip(pnr_slice, 0.0, 1.0)
        pnr = jnp.round(pnr_raw * self.pnr_max).astype(jnp.int32)

        leaf_params = {
            "n_ctrl": leaf_n_ctrl,
            "tmss_r": tmss_r,
            "us_phase": us_phase,
            "uc_theta": uc_theta,
            "uc_phi": uc_phi,
            "uc_varphi": uc_varphi,
            "disp_s": disp_s,
            "disp_c": disp_c,
            "pnr": pnr,
            "pnr_max": jnp.full((n_leaves,), self.pnr_max, dtype=jnp.int32),
        }

        # 3. Mix Nodes
        n_mix = self.nodes
        mix_params_flat = g[idx : idx + n_mix * self.PN]
        idx += n_mix * self.PN
        mix_reshaped = mix_params_flat.reshape((n_mix, self.PN))

        mix_angles = jnp.tanh(mix_reshaped[:, :3]) * (jnp.pi / 2)
        mix_source = jnp.zeros(n_mix, dtype=jnp.int32)

        # 4. Final Gaussian
        final_raw = g[idx : idx + self.F]
        idx += self.F
        r_final = jnp.tanh(final_raw[0]) * self.r_scale
        phi_final = jnp.tanh(final_raw[1]) * (jnp.pi / 2)
        varphi_final = jnp.tanh(final_raw[2]) * (jnp.pi / 2)
        disp_final = (
            jnp.tanh(final_raw[3]) * self.d_scale
            + 1j * jnp.tanh(final_raw[4]) * self.d_scale
        )

        final_gauss = {
            "r": r_final,
            "phi": phi_final,
            "varphi": varphi_final,
            "disp": disp_final,
        }

        return {
            "homodyne_x": homodyne_x,  # Array (N,)
            "homodyne_window": homodyne_window,
            "mix_params": mix_angles,
            "mix_source": mix_source,
            "leaf_active": leaf_active,
            "leaf_params": leaf_params,
            "final_gauss": final_gauss,
        }


class DesignB30Genotype(DesignB3Genotype):
    """
    Design B30: B3 + Per-Node Homodyne.
    Shared Mix Params (Continuous), Unique Homodyne.
    """

    def __init__(self, depth: int = 3, config: Dict[str, Any] = None):
        super().__init__(depth, config)
        if config is None or "hx_scale" not in config:
            self.hx_scale = 5.0

    def get_length(self, depth: int = 3) -> int:
        base_len = super().get_length(depth)
        L = 2**depth
        # Replace 1 global homodyne with (L-1) local ones
        # Diff = (L-1) - 1 = L - 2
        return base_len + (L - 2)

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        # Need to intercept 'homodyne_x' extraction which happens first in B3?
        # B3 inherits C1 for shared part? Or B1?
        # Let's check B3 hierarchy.
        # B3 does NOT inherit, it's standalone BaseGenotype subclass in valid code?
        # I need to check B3 implementation.
        # Assuming B3 decode flow starts with homodyne...

        # Strategy:
        # 1. Extract hom_x array first (like Design0).
        # 2. Extract rest of parameters using B3 length logic, but skipping the first scalar hom_x slot.
        #    BUT B3 decode likely expects g[0] to be hom_x.
        #    If I pass g[offset:] to super().decode(), B3 will read g[offset] as hom_x.
        #    I can let it read a dummy, then overwrite.
        #    BUT the lengths must match.
        #    B3 Length included 1 hom_x.
        #    My g starts with (L-1) hom_x.
        #    I can construct a synthetic genotype for B3 decode:
        #      g_b3 = [ 0.0 (dummy hom_x) ] + g[(L-1):]
        #      decoded = super().decode(g_b3)
        #      decoded['homodyne_x'] = my_parsed_hom_x_array

        n_nodes = self.nodes
        hom_x_raw = g[:n_nodes]
        homodyne_x = jnp.tanh(hom_x_raw) * self.hx_scale

        # Construct synthetic g for B3
        # B3 expects [hom_x_scalar, hom_win, shared_block...]
        # Wait, hom_win comes after hom_x in B3?
        # Let's assume standard order: hom_x, hom_win, ...
        # I need to preserve hom_win and everything else from g.
        # g structure for B30: [hom_x_array(L-1), hom_win(1), rest...]
        # g structure for B3:  [hom_x_scalar(1), hom_win(1), rest...]
        # So I take g[n_nodes:] -> this is [hom_win, rest...]
        # I prepend [0.0] -> [0.0, hom_win, rest...]
        # This matches B3 structure.

        g_rest = g[n_nodes:]
        g_synthetic = jnp.concatenate([jnp.array([0.0]), g_rest])

        decoded = super().decode(g_synthetic, cutoff)
        decoded["homodyne_x"] = homodyne_x

        return decoded


class DesignC20Genotype(DesignC2Genotype):
    """
    Design C20: C2 + Per-Node Homodyne.
    Tied Mix Params, Unique Homodyne.
    """

    def __init__(self, depth: int = 3, config: Dict[str, Any] = None):
        super().__init__(depth, config)
        if config is None or "hx_scale" not in config:
            self.hx_scale = 5.0

    def get_length(self, depth: int = 3) -> int:
        base_len = super().get_length(depth)
        L = 2**depth
        # Replace 1 global homodyne with (L-1) local ones
        return base_len + (L - 2)

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        # C2 inherits C1.
        # C1 decode: hom_x, hom_win, ...
        # Same strategy as B30.

        n_nodes = self.nodes
        hom_x_raw = g[:n_nodes]
        homodyne_x = jnp.tanh(hom_x_raw) * self.hx_scale

        g_rest = g[n_nodes:]
        g_synthetic = jnp.concatenate([jnp.array([0.0]), g_rest])

        decoded = super().decode(g_synthetic, cutoff)
        decoded["homodyne_x"] = homodyne_x

        return decoded


class DesignC2BGenotype(DesignC2Genotype):
    """Balanced version of C2 (50:50 Beam Splitters)."""

    def get_length(self, depth: int = 3) -> int:
        # C2 includes C1 part (Hom, BP, Mix, Final) + Active.
        # C1 mixed params are 3 (Tied-All: theta, phi, varphi).
        # We remove these 3.
        return super().get_length(depth) - 3

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        # C2 input g: [Hom(1), BP, Final, Active]
        # Need to inject 3 zeros into C2 structure between BP and Final.
        # Structure of C1 (which C2 uses): Hom(1) + BP + Mix(3) + Final.

        # Calculate BP (Shared Block Length)
        N_C = self.n_control
        n_uc_pairs = (N_C * (N_C - 1)) // 2
        len_uc = 2 * n_uc_pairs + N_C
        len_disp_c = 2 * N_C
        len_pnr = N_C
        BP = 1 + 1 + 1 + len_uc + 2 + len_disp_c + len_pnr

        split_idx = 1 + BP  # After Hom(1) + BP

        g_mix_zeros = jnp.zeros(3)
        g_syn = jnp.concatenate([g[:split_idx], g_mix_zeros, g[split_idx:]])

        decoded = super().decode(g_syn, cutoff)

        # Override Mix to Balanced (Time-Reversal Symmetric 50:50 BS?)
        # Standard Balanced BS: theta=pi/4, phi=0.
        mix_one = jnp.array([[jnp.pi / 4, 0.0, 0.0]])
        mix_angles = jnp.broadcast_to(mix_one, (self.nodes, 3))
        decoded["mix_params"] = mix_angles

        return decoded


class DesignC20BGenotype(DesignC20Genotype):
    """Balanced version of C20 (50:50 Beam Splitters)."""

    def get_length(self, depth: int = 3) -> int:
        # C20 inherits C2. Mix params are 3 (Tied).
        return super().get_length(depth) - 3

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        # C20 expects: [Hom(L-1), BP, Mix(3), Final, Active]
        # C20B input: [Hom(L-1), BP, Final, Active]

        # Calculate BP
        N_C = self.n_control
        n_uc_pairs = (N_C * (N_C - 1)) // 2
        len_uc = 2 * n_uc_pairs + N_C
        len_disp_c = 2 * N_C
        len_pnr = N_C
        BP = 1 + 1 + 1 + len_uc + 2 + len_disp_c + len_pnr

        # Split after Hom(L-1) + BP
        # C20 uses (L-1) nodes for homodyne x
        split_idx = self.nodes + BP

        g_mix_zeros = jnp.zeros(3)
        g_syn = jnp.concatenate([g[:split_idx], g_mix_zeros, g[split_idx:]])

        # Use super().decode() which is C20.decode -> C2.decode (via swap) logic
        decoded = super().decode(g_syn, cutoff)

        # Override Mix
        mix_one = jnp.array([[jnp.pi / 4, 0.0, 0.0]])
        mix_angles = jnp.broadcast_to(mix_one, (self.nodes, 3))
        decoded["mix_params"] = mix_angles

        return decoded


class DesignB3BGenotype(DesignB3Genotype):
    """Balanced version of B3 (50:50 Beam Splitters)."""

    def get_length(self, depth: int = 3) -> int:
        # B3 has 'nodes' mixing blocks, each 3 params.
        return super().get_length(depth) - 3 * self.nodes

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        # B3 Input: [Hom(1), Shared, Unique*L, Mix(3*nodes), Final]
        # B3B Input: [Hom(1), Shared, Unique*L, Final]

        # B3 decode structure matches definition.
        split_idx = 1 + self.Sharedv + self.leaves * self.Unique

        n_mix_params = 3 * self.nodes
        g_mix = jnp.zeros(n_mix_params)
        g_syn = jnp.concatenate([g[:split_idx], g_mix, g[split_idx:]])

        decoded = super().decode(g_syn, cutoff)

        # Override Mix
        mix_one = jnp.array([[jnp.pi / 4, 0.0, 0.0]])
        mix_angles = jnp.broadcast_to(mix_one, (self.nodes, 3))
        decoded["mix_params"] = mix_angles

        return decoded


class DesignB30BGenotype(DesignB30Genotype):
    """Balanced version of B30 (50:50 Beam Splitters)."""

    def get_length(self, depth: int = 3) -> int:
        return super().get_length(depth) - 3 * self.nodes

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        # B30 Input: [Hom(L-1), Shared, Unique*L, Mix, Final]
        # B30B Input: [Hom(L-1), Shared, Unique*L, Final]

        # Split after Hom(L-1) + Shared + Unique*L
        split_idx = self.nodes + self.Sharedv + self.leaves * self.Unique

        n_mix_params = 3 * self.nodes
        g_mix = jnp.zeros(n_mix_params)
        g_syn = jnp.concatenate([g[:split_idx], g_mix, g[split_idx:]])

        decoded = super().decode(g_syn, cutoff)

        # Override Mix
        mix_one = jnp.array([[jnp.pi / 4, 0.0, 0.0]])
        mix_angles = jnp.broadcast_to(mix_one, (self.nodes, 3))
        decoded["mix_params"] = mix_angles

        return decoded


GENOTYPE_REGISTRY = {
    "legacy": LegacyGenotype,
    "A": DesignAGenotype,
    "B1": DesignB1Genotype,
    "B2": DesignB2Genotype,
    "B3": DesignB3Genotype,
    "B30": DesignB30Genotype,
    "B3B": DesignB3BGenotype,
    "B30B": DesignB30BGenotype,
    "C1": DesignC1Genotype,
    "C2": DesignC2Genotype,
    "C20": DesignC20Genotype,
    "C2B": DesignC2BGenotype,
    "C20B": DesignC20BGenotype,
}


def get_genotype_decoder(
    name: str, depth: int = 3, config: Dict[str, Any] = None
) -> BaseGenotype:
    if name in ["0", "design0"]:
        return Design0Genotype(depth=depth, config=config)

    cls = GENOTYPE_REGISTRY.get(name)
    if not cls:
        raise ValueError(f"Unknown genotype type: {name}")
    # Legacy doesn't take depth, others do
    if name == "legacy":
        return cls(config=config)
    return cls(depth=depth, config=config)

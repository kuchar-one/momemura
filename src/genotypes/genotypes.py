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
        self.pnr_max = int(self.config.get("pnr_max", 3))
        self.active_threshold = 0.0

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
        source_raw = mix_params_reshaped[:, 3]
        mix_source = jnp.zeros(n_mix, dtype=jnp.int32)
        mix_source = jnp.where(source_raw < -0.33, 1, mix_source)
        mix_source = jnp.where(source_raw > 0.33, 2, mix_source)

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
        # P_leaf_full = 16 (active included), PN = 4, G = 1, Final = 5
        # Note: P_leaf_full is 16 now (was 17)
        self.P_leaf_full = 16
        self.PN = 4
        self.G = 1
        self.F = 5

    def get_length(self, depth: int = 3) -> int:
        L = 2**depth
        # 1(HomX) + 16*L + 4*(L-1) + 5
        # = 1 + 16L + 4L - 4 + 5 = 20L + 2
        return 20 * L + 2

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        idx = 0

        # 1. Global Homodyne X
        hom_x_raw = g[idx]
        idx += 1
        homodyne_x = jnp.tanh(hom_x_raw) * self.hx_scale
        homodyne_window = self.window

        # 2. Leaves (L * 16)
        n_leaves = self.leaves
        n_leaf_params = self.P_leaf_full
        leaves_flat = g[idx : idx + n_leaves * n_leaf_params]
        idx += n_leaves * n_leaf_params
        leaves_reshaped = leaves_flat.reshape((n_leaves, n_leaf_params))

        # Decode Leaf Block (indices 0..15)
        # 0: Active
        leaf_active = leaves_reshaped[:, 0] > 0.0

        # 1: n_ctrl
        val = leaves_reshaped[:, 1]
        leaf_n_ctrl = jnp.ones(n_leaves, dtype=jnp.int32)
        leaf_n_ctrl = jnp.where(val < -0.33, 0, leaf_n_ctrl)
        leaf_n_ctrl = jnp.where(val > 0.33, 2, leaf_n_ctrl)

        # 2: tmss_r (Single Scalar)
        tmss_r = jnp.tanh(leaves_reshaped[:, 2]) * self.r_scale  # shape (L,)

        # 3: us_phase
        us_phase = jnp.tanh(leaves_reshaped[:, 3:4]) * (jnp.pi / 2)

        # 4-7: uc_params
        uc_raw = leaves_reshaped[:, 4:8]
        uc_theta = jnp.tanh(uc_raw[:, 0:1]) * (jnp.pi / 2)
        uc_phi = jnp.tanh(uc_raw[:, 1:2]) * (jnp.pi / 2)
        uc_varphi = jnp.tanh(uc_raw[:, 2:4]) * (jnp.pi / 2)

        # 8-9: disp_s
        disp_s_raw = leaves_reshaped[:, 8:10]
        disp_s = (
            jnp.tanh(disp_s_raw[:, 0]) * self.d_scale
            + 1j * jnp.tanh(disp_s_raw[:, 1]) * self.d_scale
        )
        disp_s = disp_s[:, None]

        # 10-13: disp_c
        disp_c_raw = leaves_reshaped[:, 10:14]
        disp_c = (
            jnp.tanh(disp_c_raw[:, 0::2]) * self.d_scale
            + 1j * jnp.tanh(disp_c_raw[:, 1::2]) * self.d_scale
        )

        # 14-15: pnr
        pnr_raw = jnp.clip(leaves_reshaped[:, 14:16], 0.0, 1.0)
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
        }

        # 3. Mix Nodes
        n_mix = self.nodes
        mix_params_flat = g[idx : idx + n_mix * self.PN]
        idx += n_mix * self.PN
        mix_reshaped = mix_params_flat.reshape((n_mix, self.PN))

        mix_angles = jnp.tanh(mix_reshaped[:, :3]) * (jnp.pi / 2)
        src_raw = mix_reshaped[:, 3]
        mix_source = jnp.zeros(n_mix, dtype=jnp.int32)
        mix_source = jnp.where(src_raw < -0.33, 1, mix_source)
        mix_source = jnp.where(src_raw > 0.33, 2, mix_source)

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
        # BP = 15, PN = 4, G = 1, F = 5
        self.BP = 15
        self.PN = 4
        self.G = 1
        self.F = 5

    def get_length(self, depth: int = 3) -> int:
        L = 2**depth
        # 1 + 15 + 4*(L-1) + 5 = 21 + 4L - 4 = 17 + 4L
        return 17 + 4 * L

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        idx = 0
        hom_x_raw = g[idx]
        idx += 1
        homodyne_x = jnp.tanh(hom_x_raw) * self.hx_scale
        homodyne_window = self.window

        # Shared Block (15 params)
        bp_raw = g[idx : idx + self.BP]
        idx += self.BP

        # 0: n_ctrl
        val = bp_raw[0]
        n_ctrl_int = 1
        n_ctrl_int = jnp.where(val < -0.33, 0, n_ctrl_int)
        n_ctrl_int = jnp.where(val > 0.33, 2, n_ctrl_int)

        # 1: tmss_r (single)
        tmss_r_val = jnp.tanh(bp_raw[1]) * self.r_scale

        # 2: us_phase (index shift from here)
        us_phase_val = jnp.tanh(bp_raw[2:3]) * (jnp.pi / 2)

        # 3-6: uc_params
        uc_params_val = jnp.tanh(bp_raw[3:7]) * (jnp.pi / 2)
        uc_theta_val = uc_params_val[0:1]
        uc_phi_val = uc_params_val[1:2]
        uc_varphi_val = uc_params_val[2:4]

        # 7-8: disp_s
        disp_s_raw = bp_raw[7:9]
        disp_s_val = (
            jnp.tanh(disp_s_raw[0]) * self.d_scale
            + 1j * jnp.tanh(disp_s_raw[1]) * self.d_scale
        )
        disp_s_val = disp_s_val[None]

        # 9-12: disp_c
        disp_c_raw = bp_raw[9:13]
        disp_c_val = (
            jnp.tanh(disp_c_raw[0::2]) * self.d_scale
            + 1j * jnp.tanh(disp_c_raw[1::2]) * self.d_scale
        )

        # 13-14: pnr (end of new BP=15)
        pnr_raw = jnp.clip(bp_raw[13:15], 0.0, 1.0)
        pnr_val = jnp.round(pnr_raw * self.pnr_max).astype(jnp.int32)

        # Broadcast
        n_leaves = self.leaves
        leaf_params = {
            "n_ctrl": jnp.broadcast_to(n_ctrl_int, (n_leaves,)),
            "tmss_r": jnp.broadcast_to(tmss_r_val, (n_leaves,)),  # 1D
            "us_phase": jnp.broadcast_to(us_phase_val, (n_leaves, 1)),
            "uc_theta": jnp.broadcast_to(uc_theta_val, (n_leaves, 1)),
            "uc_phi": jnp.broadcast_to(uc_phi_val, (n_leaves, 1)),
            "uc_varphi": jnp.broadcast_to(uc_varphi_val, (n_leaves, 2)),
            "disp_s": jnp.broadcast_to(disp_s_val, (n_leaves, 1)),
            "disp_c": jnp.broadcast_to(disp_c_val, (n_leaves, 2)),
            "pnr": jnp.broadcast_to(pnr_val, (n_leaves, 2)),
        }
        leaf_active = jnp.ones(n_leaves, dtype=bool)

        # Mix Params
        n_mix = self.nodes
        mix_params_flat = g[idx : idx + n_mix * self.PN]
        idx += n_mix * self.PN
        mix_reshaped = mix_params_flat.reshape((n_mix, self.PN))
        mix_angles = jnp.tanh(mix_reshaped[:, :3]) * (jnp.pi / 2)

        src_raw = mix_reshaped[:, 3]
        mix_source = jnp.zeros(n_mix, dtype=jnp.int32)
        mix_source = jnp.where(src_raw < -0.33, 1, mix_source)
        mix_source = jnp.where(src_raw > 0.33, 2, mix_source)

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


class DesignB2Genotype(DesignB1Genotype):
    """
    Design B2: Tied-leaf with per-leaf active flags.
    Adds L booleans.
    """

    def get_length(self, depth: int = 3) -> int:
        L = 2**depth
        # B1 (17+4L) + L = 17 + 5L
        return 17 + 5 * L  # Correct matches spec

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        # Copied from B1 logic then add Active at end
        idx = 0
        hom_x_raw = g[idx]
        idx += 1
        homodyne_x = jnp.tanh(hom_x_raw) * self.hx_scale
        homodyne_window = self.window

        bp_raw = g[idx : idx + self.BP]
        idx += self.BP

        # Block params (same indices as B1)
        val = bp_raw[0]
        n_ctrl_int = 1
        n_ctrl_int = jnp.where(val < -0.33, 0, n_ctrl_int)
        n_ctrl_int = jnp.where(val > 0.33, 2, n_ctrl_int)

        tmss_r_val = jnp.tanh(bp_raw[1]) * self.r_scale
        us_phase_val = jnp.tanh(bp_raw[2:3]) * (jnp.pi / 2)
        uc_params_val = jnp.tanh(bp_raw[3:7]) * (jnp.pi / 2)
        uc_theta_val = uc_params_val[0:1]
        uc_phi_val = uc_params_val[1:2]
        uc_varphi_val = uc_params_val[2:4]

        disp_s_raw = bp_raw[7:9]
        disp_s_val = (
            jnp.tanh(disp_s_raw[0]) * self.d_scale
            + 1j * jnp.tanh(disp_s_raw[1]) * self.d_scale
        )
        disp_s_val = disp_s_val[None]

        disp_c_raw = bp_raw[9:13]
        disp_c_val = (
            jnp.tanh(disp_c_raw[0::2]) * self.d_scale
            + 1j * jnp.tanh(disp_c_raw[1::2]) * self.d_scale
        )

        pnr_raw = jnp.clip(bp_raw[13:15], 0.0, 1.0)
        pnr_val = jnp.round(pnr_raw * self.pnr_max).astype(jnp.int32)

        n_leaves = self.leaves
        leaf_params = {
            "n_ctrl": jnp.broadcast_to(n_ctrl_int, (n_leaves,)),
            "tmss_r": jnp.broadcast_to(tmss_r_val, (n_leaves,)),
            "us_phase": jnp.broadcast_to(us_phase_val, (n_leaves, 1)),
            "uc_theta": jnp.broadcast_to(uc_theta_val, (n_leaves, 1)),
            "uc_phi": jnp.broadcast_to(uc_phi_val, (n_leaves, 1)),
            "uc_varphi": jnp.broadcast_to(uc_varphi_val, (n_leaves, 2)),
            "disp_s": jnp.broadcast_to(disp_s_val, (n_leaves, 1)),
            "disp_c": jnp.broadcast_to(disp_c_val, (n_leaves, 2)),
            "pnr": jnp.broadcast_to(pnr_val, (n_leaves, 2)),
        }

        # Mix params
        n_mix = self.nodes
        mix_params_flat = g[idx : idx + n_mix * self.PN]
        idx += n_mix * self.PN
        mix_reshaped = mix_params_flat.reshape((n_mix, self.PN))
        mix_angles = jnp.tanh(mix_reshaped[:, :3]) * (jnp.pi / 2)

        src_raw = mix_reshaped[:, 3]
        mix_source = jnp.zeros(n_mix, dtype=jnp.int32)
        mix_source = jnp.where(src_raw < -0.33, 1, mix_source)
        mix_source = jnp.where(src_raw > 0.33, 2, mix_source)

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

        # Active Flags
        active_raw = g[idx : idx + n_leaves]
        idx += n_leaves
        leaf_active = active_raw > self.active_threshold

        return {
            "homodyne_x": homodyne_x,
            "homodyne_window": homodyne_window,
            "mix_params": mix_angles,
            "mix_source": mix_source,
            "leaf_active": leaf_active,
            "leaf_params": leaf_params,
            "final_gauss": final_gauss,
        }


class DesignC1Genotype(BaseGenotype):
    """
    Design C1: Tied-all, no active flags.
    BP=15, PN=4, G=1, F=5.
    Total = 1 + 15 + 4 + 5 = 25.
    """

    def __init__(self, depth: int = 3, config: Dict[str, Any] = None):
        super().__init__(depth, config)
        self.leaves = 2**depth
        self.nodes = self.leaves - 1
        self.BP = 15
        self.PN = 4
        self.G = 1
        self.F = 5

    def get_length(self, depth: int = 3) -> int:
        return 25

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        idx = 0
        hom_x_raw = g[idx]
        idx += 1
        homodyne_x = jnp.tanh(hom_x_raw) * self.hx_scale
        homodyne_window = self.window

        # Shared Block (15)
        bp_raw = g[idx : idx + self.BP]
        idx += self.BP

        val = bp_raw[0]
        n_ctrl_int = 1
        n_ctrl_int = jnp.where(val < -0.33, 0, n_ctrl_int)
        n_ctrl_int = jnp.where(val > 0.33, 2, n_ctrl_int)

        tmss_r_val = jnp.tanh(bp_raw[1]) * self.r_scale  # single
        us_phase_val = jnp.tanh(bp_raw[2:3]) * (jnp.pi / 2)
        uc_params_val = jnp.tanh(bp_raw[3:7]) * (jnp.pi / 2)
        uc_theta_val = uc_params_val[0:1]
        uc_phi_val = uc_params_val[1:2]
        uc_varphi_val = uc_params_val[2:4]

        disp_s_raw = bp_raw[7:9]
        disp_s_val = (
            jnp.tanh(disp_s_raw[0]) * self.d_scale
            + 1j * jnp.tanh(disp_s_raw[1]) * self.d_scale
        )
        disp_s_val = disp_s_val[None]

        disp_c_raw = bp_raw[9:13]
        disp_c_val = (
            jnp.tanh(disp_c_raw[0::2]) * self.d_scale
            + 1j * jnp.tanh(disp_c_raw[1::2]) * self.d_scale
        )

        pnr_raw = jnp.clip(bp_raw[13:15], 0.0, 1.0)
        pnr_val = jnp.round(pnr_raw * self.pnr_max).astype(jnp.int32)

        n_leaves = self.leaves
        leaf_params = {
            "n_ctrl": jnp.broadcast_to(n_ctrl_int, (n_leaves,)),
            "tmss_r": jnp.broadcast_to(tmss_r_val, (n_leaves,)),
            "us_phase": jnp.broadcast_to(us_phase_val, (n_leaves, 1)),
            "uc_theta": jnp.broadcast_to(uc_theta_val, (n_leaves, 1)),
            "uc_phi": jnp.broadcast_to(uc_phi_val, (n_leaves, 1)),
            "uc_varphi": jnp.broadcast_to(uc_varphi_val, (n_leaves, 2)),
            "disp_s": jnp.broadcast_to(disp_s_val, (n_leaves, 1)),
            "disp_c": jnp.broadcast_to(disp_c_val, (n_leaves, 2)),
            "pnr": jnp.broadcast_to(pnr_val, (n_leaves, 2)),
        }
        leaf_active = jnp.ones(n_leaves, dtype=bool)

        # Shared Mix Node (4)
        mix_raw = g[idx : idx + self.PN]
        idx += self.PN
        mix_angles = jnp.tanh(mix_raw[:3]) * (jnp.pi / 2)
        src_raw = mix_raw[3]
        mix_source_val = 0
        mix_source_val = jnp.where(src_raw < -0.33, 1, mix_source_val)
        mix_source_val = jnp.where(src_raw > 0.33, 2, mix_source_val)

        n_mix = self.nodes
        mix_params_all = jnp.broadcast_to(mix_angles, (n_mix, 3))
        mix_source_all = jnp.broadcast_to(mix_source_val, (n_mix,))

        # Final Gaussian (5)
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
            "mix_params": mix_params_all,
            "mix_source": mix_source_all,
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
        L = 2**depth
        return 25 + L

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        result = super().decode(g, cutoff)

        # C1 uses 25 genes.
        # C2 appends L actives.
        n_leaves = self.leaves
        actives_raw = g[25 : 25 + n_leaves]
        leaf_active = actives_raw > self.active_threshold

        result["leaf_active"] = leaf_active
        return result


GENOTYPE_REGISTRY = {
    "legacy": LegacyGenotype,
    "A": DesignAGenotype,
    "B1": DesignB1Genotype,
    "B2": DesignB2Genotype,
    "C1": DesignC1Genotype,
    "C2": DesignC2Genotype,
}


def get_genotype_decoder(
    name: str, depth: int = 3, config: Dict[str, Any] = None
) -> BaseGenotype:
    cls = GENOTYPE_REGISTRY.get(name)
    if not cls:
        raise ValueError(f"Unknown genotype type: {name}")
    # Legacy doesn't take depth, others do
    if name == "legacy":
        return cls(config=config)
    return cls(depth=depth, config=config)

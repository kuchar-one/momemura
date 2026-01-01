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
    "modes": 3,  # Default to 3 modes (1 Signal + 2 Controls)
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

        # Dynamic Modes (1 Signal + N-1 Controls)
        self.n_modes = int(self.config.get("modes", 3))
        self.n_control = max(1, self.n_modes - 1)

        # Helper counts for General Gaussian
        self.gg_len = self._get_general_gaussian_length(self.n_modes)
        self.pnr_len = self.n_control

        self.leaves = 2**depth
        self.nodes = self.leaves - 1

    @abstractmethod
    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_length(self, depth: int = 3) -> int:
        pass

    def _pad_genotype(self, g: jnp.ndarray) -> jnp.ndarray:
        return g

    def _get_general_gaussian_length(self, n_modes: int) -> int:
        len_r = n_modes
        len_phases = n_modes * n_modes
        len_disp = 2 * n_modes
        return len_r + len_phases + len_disp

    def _decode_general_gaussian(
        self, flat_params: jnp.ndarray, n_modes: int
    ) -> Dict[str, Any]:
        idx = 0
        r_raw = flat_params[idx : idx + n_modes]
        idx += n_modes
        r = jnp.tanh(r_raw) * self.r_scale

        len_phases = n_modes * n_modes
        phases_raw = flat_params[idx : idx + len_phases]
        idx += len_phases
        # Map to [0, 2pi]
        phases = (jnp.tanh(phases_raw) + 1.0) * jnp.pi

        len_disp = 2 * n_modes
        disp_raw = flat_params[idx : idx + len_disp]
        idx += len_disp
        disp = (
            jnp.tanh(disp_raw[0::2]) * self.d_scale
            + 1j * jnp.tanh(disp_raw[1::2]) * self.d_scale
        )
        return {"r": r, "phases": phases, "disp": disp}


# --- Design A & 0 (Per-Leaf Unique) ---


class DesignAGenotype(BaseGenotype):
    """Design A: Unique per leaf."""

    def __init__(self, depth: int = 3, config: Dict[str, Any] = None):
        super().__init__(depth, config)
        # Leaf: Active(1) + NCtrl(1) + PNR(N_C) + GG
        self.P_leaf_full = 1 + 1 + self.pnr_len + self.gg_len
        self.PN = 3
        self.F = 5

    def get_length(self, depth: int = 3) -> int:
        L = 2**depth
        # Global Hom(1) + Leaves + Mix(Nodes*3) + Final(5)
        return 1 + L * self.P_leaf_full + (L - 1) * self.PN + self.F

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        idx = 0
        hom_x = jnp.tanh(g[idx]) * self.hx_scale
        idx += 1
        hom_win = self.window

        n_leaves = self.leaves
        P_leaf = self.P_leaf_full
        leaves_flat = g[idx : idx + n_leaves * P_leaf]
        idx += n_leaves * P_leaf
        leaves_reshaped = leaves_flat.reshape((n_leaves, P_leaf))

        leaf_active = leaves_reshaped[:, 0] > 0.0

        n_ctrl_raw = leaves_reshaped[:, 1]
        leaf_n_ctrl = jnp.round((n_ctrl_raw + 1.0) / 2.0 * self.n_control).astype(
            jnp.int32
        )
        leaf_n_ctrl = jnp.clip(leaf_n_ctrl, 0, self.n_control)

        pnr_slice = leaves_reshaped[:, 2 : 2 + self.pnr_len]
        pnr = jnp.round(jnp.clip(pnr_slice, 0, 1) * self.pnr_max).astype(jnp.int32)

        gg_slice = leaves_reshaped[:, 2 + self.pnr_len :]
        r = jnp.tanh(gg_slice[:, : self.n_modes]) * self.r_scale

        ph_start = self.n_modes
        ph_len = self.n_modes**2
        phases = (jnp.tanh(gg_slice[:, ph_start : ph_start + ph_len]) + 1.0) * jnp.pi

        d_start = ph_start + ph_len
        d_slice = gg_slice[:, d_start:]
        disp = (
            jnp.tanh(d_slice[:, 0::2]) * self.d_scale
            + 1j * jnp.tanh(d_slice[:, 1::2]) * self.d_scale
        )

        leaf_params = {
            "n_ctrl": leaf_n_ctrl,
            "pnr": pnr,
            "r": r,
            "phases": phases,
            "disp": disp,
            "pnr_max": jnp.full((n_leaves,), self.pnr_max, dtype=jnp.int32),
        }

        n_mix = self.nodes
        mix_params_flat = g[idx : idx + n_mix * self.PN]
        idx += n_mix * self.PN
        mix_reshaped = mix_params_flat.reshape((n_mix, self.PN))
        mix_angles = jnp.tanh(mix_reshaped[:, :3]) * (jnp.pi / 2)
        mix_source = jnp.zeros(n_mix, dtype=jnp.int32)

        final_raw = g[idx : idx + self.F]
        r_final = jnp.tanh(final_raw[0]) * self.r_scale
        phi_final = jnp.tanh(final_raw[1]) * (jnp.pi / 2)
        varphi_final = jnp.tanh(final_raw[2]) * (jnp.pi / 2)
        disp_final = (
            jnp.tanh(final_raw[3]) * self.d_scale
            + 1j * jnp.tanh(final_raw[4]) * self.d_scale
        )

        return {
            "homodyne_x": hom_x,
            "homodyne_window": hom_win,
            "mix_params": mix_angles,
            "mix_source": mix_source,
            "leaf_active": leaf_active,
            "leaf_params": leaf_params,
            "final_gauss": {
                "r": r_final,
                "phi": phi_final,
                "varphi": varphi_final,
                "disp": disp_final,
            },
        }


class Design0Genotype(DesignAGenotype):
    """Design 0: Same as A but Per-Node Homodyne (L-1 params) instead of global."""

    def get_length(self, depth: int = 3) -> int:
        base = super().get_length(depth)
        # remove 1 global, add (L-1) local
        return base - 1 + (self.nodes)

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        idx = 0
        # Per-Node Homodyne X (L-1)
        hom_x_raw = g[idx : idx + self.nodes]
        idx += self.nodes
        hom_x = jnp.tanh(hom_x_raw) * self.hx_scale

        base_res = super().decode(g[idx:], cutoff)

        # Override hom_x
        base_res["homodyne_x"] = hom_x
        return base_res


# --- Design B Variants (Tied-Leaf / Semi-Tied) ---


class DesignB1Genotype(BaseGenotype):
    """Design B1: Tied-Leaf (No Active). Shared GG + PNR + NCtrl."""

    def __init__(self, depth: int = 3, config: Dict[str, Any] = None):
        super().__init__(depth, config)
        self.BP = 1 + self.pnr_len + self.gg_len
        self.PN = 3
        self.F = 5

    def get_length(self, depth: int = 3) -> int:
        L = 2**depth
        return 1 + self.BP + (L - 1) * self.PN + self.F

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        idx = 0
        hom_x = jnp.tanh(g[idx]) * self.hx_scale
        idx += 1
        hom_win = self.window

        # Shared Block
        bp_raw = g[idx : idx + self.BP]
        idx += self.BP

        b_idx = 0
        n_ctrl = jnp.round((bp_raw[b_idx] + 1.0) / 2.0 * self.n_control).astype(
            jnp.int32
        )
        n_ctrl = jnp.clip(n_ctrl, 0, self.n_control)
        b_idx += 1

        pnr = jnp.round(
            jnp.clip(bp_raw[b_idx : b_idx + self.pnr_len], 0, 1) * self.pnr_max
        ).astype(jnp.int32)
        b_idx += self.pnr_len

        gg = self._decode_general_gaussian(bp_raw[b_idx:], self.n_modes)

        L = self.leaves
        leaf_params = {
            "n_ctrl": jnp.broadcast_to(n_ctrl, (L,)),
            "pnr": jnp.broadcast_to(pnr, (L, self.pnr_len)),
            "r": jnp.broadcast_to(gg["r"], (L, self.n_modes)),
            "phases": jnp.broadcast_to(gg["phases"], (L, self.n_modes**2)),
            "disp": jnp.broadcast_to(gg["disp"], (L, self.n_modes)),
            "pnr_max": jnp.full((L,), self.pnr_max, dtype=jnp.int32),
        }
        leaf_active = jnp.ones(L, dtype=bool)

        n_mix = self.nodes
        mix_params_flat = g[idx : idx + n_mix * self.PN]
        idx += n_mix * self.PN
        mix_angles = jnp.tanh(mix_params_flat.reshape((n_mix, self.PN))[:, :3]) * (
            jnp.pi / 2
        )
        mix_source = jnp.zeros(n_mix, dtype=jnp.int32)

        final_raw = g[idx : idx + self.F]
        r_final = jnp.tanh(final_raw[0]) * self.r_scale
        phi_final = jnp.tanh(final_raw[1]) * (jnp.pi / 2)
        varphi_final = jnp.tanh(final_raw[2]) * (jnp.pi / 2)
        disp_final = (
            jnp.tanh(final_raw[3]) * self.d_scale
            + 1j * jnp.tanh(final_raw[4]) * self.d_scale
        )

        return {
            "homodyne_x": hom_x,
            "homodyne_window": hom_win,
            "mix_params": mix_angles,
            "mix_source": mix_source,
            "leaf_active": leaf_active,
            "leaf_params": leaf_params,
            "final_gauss": {
                "r": r_final,
                "phi": phi_final,
                "varphi": varphi_final,
                "disp": disp_final,
            },
        }


class DesignB2Genotype(DesignB1Genotype):
    """Design B2: Tied-Leaf + L Active Flags."""

    def get_length(self, depth: int = 3) -> int:
        return super().get_length(depth) + 2**depth

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        b1_len = super().get_length(self.depth)
        decoded = super().decode(g[:b1_len], cutoff)
        active_raw = g[b1_len : b1_len + self.leaves]
        decoded["leaf_active"] = active_raw > self.active_threshold
        return decoded


class DesignB3Genotype(BaseGenotype):
    """Design B3: Semi-Tied. Shared Continuous, Unique Discrete."""

    def __init__(self, depth: int = 3, config: Dict[str, Any] = None):
        super().__init__(depth, config)
        self.Sharedv = self.gg_len
        self.Unique = 1 + 1 + self.pnr_len
        self.PN = 3
        self.F = 5

    def get_length(self, depth: int = 3) -> int:
        L = 2**depth
        return 1 + self.Sharedv + L * self.Unique + (L - 1) * self.PN + self.F

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        idx = 0
        hom_x = jnp.tanh(g[idx]) * self.hx_scale
        idx += 1
        hom_win = self.window

        shared_raw = g[idx : idx + self.Sharedv]
        idx += self.Sharedv
        gg = self._decode_general_gaussian(shared_raw, self.n_modes)

        L = self.leaves
        unique_total = L * self.Unique
        unique_raw = g[idx : idx + unique_total]
        idx += unique_total
        unique_reshaped = unique_raw.reshape((L, self.Unique))

        leaf_active = unique_reshaped[:, 0] > 0.0

        n_ctrl_raw = unique_reshaped[:, 1]
        leaf_n_ctrl = jnp.round((n_ctrl_raw + 1.0) / 2.0 * self.n_control).astype(
            jnp.int32
        )
        leaf_n_ctrl = jnp.clip(leaf_n_ctrl, 0, self.n_control)

        pnr_raw = jnp.clip(unique_reshaped[:, 2:], 0, 1)
        leaf_pnr = jnp.round(pnr_raw * self.pnr_max).astype(jnp.int32)

        leaf_params = {
            "n_ctrl": leaf_n_ctrl,
            "pnr": leaf_pnr,
            "r": jnp.broadcast_to(gg["r"], (L, self.n_modes)),
            "phases": jnp.broadcast_to(gg["phases"], (L, self.n_modes**2)),
            "disp": jnp.broadcast_to(gg["disp"], (L, self.n_modes)),
            "pnr_max": jnp.full((L,), self.pnr_max, dtype=jnp.int32),
        }

        n_mix = self.nodes
        mix_params_flat = g[idx : idx + n_mix * self.PN]
        idx += n_mix * self.PN
        mix_angles = jnp.tanh(mix_params_flat.reshape((n_mix, self.PN))[:, :3]) * (
            jnp.pi / 2
        )
        mix_source = jnp.zeros(n_mix, dtype=jnp.int32)

        final_raw = g[idx : idx + self.F]
        r_final = jnp.tanh(final_raw[0]) * self.r_scale
        phi_final = jnp.tanh(final_raw[1]) * (jnp.pi / 2)
        varphi_final = jnp.tanh(final_raw[2]) * (jnp.pi / 2)
        disp_final = (
            jnp.tanh(final_raw[3]) * self.d_scale
            + 1j * jnp.tanh(final_raw[4]) * self.d_scale
        )

        return {
            "homodyne_x": hom_x,
            "homodyne_window": hom_win,
            "mix_params": mix_angles,
            "mix_source": mix_source,
            "leaf_active": leaf_active,
            "leaf_params": leaf_params,
            "final_gauss": {
                "r": r_final,
                "phi": phi_final,
                "varphi": varphi_final,
                "disp": disp_final,
            },
        }


class DesignB30Genotype(DesignB3Genotype):
    """Design B30: B3 + Per-Node Homodyne."""

    def get_length(self, depth: int = 3) -> int:
        base = super().get_length(depth)
        return base - 1 + self.nodes

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        idx = 0
        hom_x_raw = g[idx : idx + self.nodes]
        idx += self.nodes
        hom_x = jnp.tanh(hom_x_raw) * self.hx_scale

        base_res = super().decode(g[idx:], cutoff)
        base_res["homodyne_x"] = hom_x
        return base_res


class DesignB3BGenotype(DesignB3Genotype):
    """Design B3B: B3 + Balanced Mixing (Fixed)."""

    def __init__(self, depth: int = 3, config: Dict[str, Any] = None):
        super().__init__(depth, config)
        self.PN = 0  # No mix params

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        res = super().decode(g, cutoff)
        # Override random/zero mix angles with Balanced
        # Balanced: Theta=pi/4, Phi=0, Varphi=0
        nodes = self.nodes
        mix_params = jnp.zeros((nodes, 3))
        mix_params = mix_params.at[:, 0].set(jnp.pi / 4)
        res["mix_params"] = mix_params
        return res


class DesignB30BGenotype(DesignB30Genotype):
    """Design B30B: B30 + Balanced Mixing."""

    def __init__(self, depth: int = 3, config: Dict[str, Any] = None):
        super().__init__(depth, config)
        self.PN = 0

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        res = super().decode(g, cutoff)
        nodes = self.nodes
        mix_params = jnp.zeros((nodes, 3))
        mix_params = mix_params.at[:, 0].set(jnp.pi / 4)
        res["mix_params"] = mix_params
        return res


# --- Design C Variants (Tied-All) ---


class DesignC1Genotype(BaseGenotype):
    """Design C1: Tied-All (No Active). Shared GG+PNR+NCtrl + Shared Mix."""

    def __init__(self, depth: int = 3, config: Dict[str, Any] = None):
        super().__init__(depth, config)
        self.BP = 1 + self.pnr_len + self.gg_len
        self.PN = 3  # Shared 1 set
        self.F = 5

    def get_length(self, depth: int = 3) -> int:
        return 1 + self.BP + self.PN + self.F

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        idx = 0
        hom_x = jnp.tanh(g[idx]) * self.hx_scale
        idx += 1
        hom_win = self.window

        # Shared Leaf
        bp_raw = g[idx : idx + self.BP]
        idx += self.BP

        b_idx = 0
        n_ctrl = jnp.round((bp_raw[b_idx] + 1.0) / 2.0 * self.n_control).astype(
            jnp.int32
        )
        n_ctrl = jnp.clip(n_ctrl, 0, self.n_control)
        b_idx += 1
        pnr = jnp.round(
            jnp.clip(bp_raw[b_idx : b_idx + self.pnr_len], 0, 1) * self.pnr_max
        ).astype(jnp.int32)
        b_idx += self.pnr_len
        gg = self._decode_general_gaussian(bp_raw[b_idx:], self.n_modes)

        L = self.leaves
        leaf_params = {
            "n_ctrl": jnp.broadcast_to(n_ctrl, (L,)),
            "pnr": jnp.broadcast_to(pnr, (L, self.pnr_len)),
            "r": jnp.broadcast_to(gg["r"], (L, self.n_modes)),
            "phases": jnp.broadcast_to(gg["phases"], (L, self.n_modes**2)),
            "disp": jnp.broadcast_to(gg["disp"], (L, self.n_modes)),
            "pnr_max": jnp.full((L,), self.pnr_max, dtype=jnp.int32),
        }
        leaf_active = jnp.ones(L, dtype=bool)

        # Shared Mix
        mix_raw = g[idx : idx + self.PN]
        idx += self.PN
        mix_single = jnp.tanh(mix_raw) * (jnp.pi / 2)
        mix_angles = jnp.broadcast_to(mix_single, (self.nodes, 3))
        mix_source = jnp.zeros(self.nodes, dtype=jnp.int32)

        final_raw = g[idx : idx + self.F]
        r_final = jnp.tanh(final_raw[0]) * self.r_scale
        phi_final = jnp.tanh(final_raw[1]) * (jnp.pi / 2)
        varphi_final = jnp.tanh(final_raw[2]) * (jnp.pi / 2)
        disp_final = (
            jnp.tanh(final_raw[3]) * self.d_scale
            + 1j * jnp.tanh(final_raw[4]) * self.d_scale
        )

        return {
            "homodyne_x": hom_x,
            "homodyne_window": hom_win,
            "mix_params": mix_angles,
            "mix_source": mix_source,
            "leaf_active": leaf_active,
            "leaf_params": leaf_params,
            "final_gauss": {
                "r": r_final,
                "phi": phi_final,
                "varphi": varphi_final,
                "disp": disp_final,
            },
        }


class DesignC2Genotype(DesignC1Genotype):
    """Design C2: Tied-All + L Active Flags."""

    def get_length(self, depth: int = 3) -> int:
        return super().get_length(depth) + 2**depth

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        base_len = super().get_length(self.depth)
        decoded = super().decode(g[:base_len], cutoff)
        active_raw = g[base_len : base_len + self.leaves]
        decoded["leaf_active"] = active_raw > self.active_threshold
        return decoded


class DesignC2BGenotype(DesignC2Genotype):
    """Design C2B: C2 + Balanced Mixing."""

    def __init__(self, depth: int = 3, config: Dict[str, Any] = None):
        super().__init__(depth, config)
        self.PN = 0  # No mix params

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        res = super().decode(g, cutoff)
        mix_params = jnp.zeros((self.nodes, 3))
        mix_params = mix_params.at[:, 0].set(jnp.pi / 4)
        res["mix_params"] = mix_params
        return res


class DesignC20Genotype(DesignC2Genotype):
    """Design C20: C2 + Per-Node Homodyne."""

    def get_length(self, depth: int = 3) -> int:
        return super().get_length(depth) - 1 + self.nodes

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        idx = 0
        hom_x_raw = g[idx : idx + self.nodes]
        idx += self.nodes
        hom_x = jnp.tanh(hom_x_raw) * self.hx_scale

        base_res = super().decode(g[idx:], cutoff)
        base_res["homodyne_x"] = hom_x
        return base_res


class DesignC20BGenotype(DesignC20Genotype):
    """Design C20B: C20 + Balanced Mixing."""

    def __init__(self, depth: int = 3, config: Dict[str, Any] = None):
        super().__init__(depth, config)
        self.PN = 0

    def decode(self, g: jnp.ndarray, cutoff: int) -> Dict[str, Any]:
        res = super().decode(g, cutoff)
        mix_params = jnp.zeros((self.nodes, 3))
        mix_params = mix_params.at[:, 0].set(jnp.pi / 4)
        res["mix_params"] = mix_params
        return res


def get_genotype_decoder(
    name: str, depth: int = 3, config: Dict[str, Any] = None
) -> BaseGenotype:
    decoders = {
        "A": DesignAGenotype,
        "0": Design0Genotype,
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
    if name not in decoders:
        raise ValueError(f"Unknown genotype: {name}")
    return decoders[name](depth, config)

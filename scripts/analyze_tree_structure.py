#!/usr/bin/env python3
"""Decode the tree-level structure of the nine representative NG-campaign
preparations (cheap/knee/champion per target) and tabulate what the search
actually discovered, in breeding-protocol terms:

  - active leaves, per-leaf PNR patterns (tied continuous hardware by design)
  - which mixing nodes fire (both children active), their beam-splitter angle
    theta (transmittance cos^2 theta), phases, homodyne offset x0 and window
  - final Gaussian correction
  - pass-through bookkeeping, so the effective circuit is a chain/cascade view

Node ordering follows src/simulation/jax/composer.py: layer-sequential
(layer 1 = adjacent leaf pairs in order, then layer 2, ...). Mix-or-pass rule:
both children active -> mix; exactly one active -> pass it through (homodyne
and BS of that node unused); none -> inactive.

Usage: JAX_ENABLE_X64=1 python scripts/analyze_tree_structure.py
Reads pick provenance from the thesis cache (NG_DATA env or default path).
"""
import os, sys, json, math
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from src.genotypes.genotypes import get_genotype_decoder
from rescore_all_experiments import load_run_arrays

NG_DATA = os.environ.get("NG_DATA", os.path.expanduser(
    "~/Nextcloud/vojtech/writing/mgr/scripts/ng_results_data.json"))
DB = 10*math.log10(math.e**2)

def decode_pick(rec):
    run = rec["run"].split("/")[-1]
    base = os.path.join(REPO, "output/experiments", rec["group"], run)
    gens, _f, _d = load_run_arrays(os.path.join(base, "results.pkl"))
    cfg = json.load(open(os.path.join(base, "config.json")))
    gens = np.asarray(gens).reshape(-1, np.asarray(gens).shape[-1])
    dec = get_genotype_decoder(cfg.get("genotype"), depth=int(cfg.get("depth") or 3),
                               config=cfg)
    params = dec.decode(jnp.asarray(gens[rec["cell"]].astype(np.float32)),
                        int(cfg.get("cutoff") or 30))
    return params, cfg

def analyze(params):
    active = np.asarray(params["leaf_active"]).astype(bool)
    n_leaves = active.shape[0]
    depth = int(math.log2(n_leaves))
    n_ctrl = np.asarray(params["leaf_params"]["n_ctrl"])
    pnr = np.asarray(params["leaf_params"]["pnr"])
    mix = np.asarray(params["mix_params"], float)      # (n_nodes, 3) theta,phi,varphi
    hx = np.asarray(params["homodyne_x"], float).ravel()
    win = float(np.asarray(params["homodyne_window"]))
    r = np.asarray(params["leaf_params"]["r"], float)[0]      # tied across leaves
    disp = np.asarray(params["leaf_params"]["disp"], complex)[0]
    fin = {k: np.asarray(v, float).ravel().tolist()
           for k, v in params["final_gauss"].items()} if isinstance(
               params.get("final_gauss"), dict) else np.asarray(
               params.get("final_gauss")).ravel().tolist()

    leaves = [dict(idx=i, pnr=[int(x) for x in pnr[i][:max(1, int(n_ctrl[i]))]],
                   n_ctrl=int(n_ctrl[i]))
              for i in range(n_leaves) if active[i]]

    # propagate activity through the layer-sequential node list
    events = []
    layer_states = [(bool(active[i]), f"L{i}") for i in range(n_leaves)]
    node_idx = 0
    layer = 1
    while len(layer_states) > 1:
        nxt = []
        for j in range(0, len(layer_states), 2):
            (aA, tagA), (aB, tagB) = layer_states[j], layer_states[j+1]
            th, ph, vph = mix[node_idx]
            if aA and aB:
                events.append(dict(
                    node=node_idx, layer=layer, left=tagA, right=tagB,
                    theta_deg=math.degrees(th), T=math.cos(th)**2,
                    phi=ph, varphi=vph, hom_x0=float(hx[node_idx]), window=win))
                nxt.append((True, f"N{node_idx}[{tagA}+{tagB}]"))
            elif aA or aB:
                nxt.append((True, tagA if aA else tagB))
            else:
                nxt.append((False, "-"))
            node_idx += 1
        layer_states = nxt
        layer += 1
    return dict(depth=depth, leaves=leaves, events=events,
                leaf_r_db=[float(x*DB) for x in r],
                leaf_disp=[[float(disp[k].real), float(disp[k].imag)]
                           for k in range(len(disp))],
                final_gauss=fin, window=win)

def main():
    D = json.load(open(NG_DATA))
    out = {}
    for t in ("plus", "H", "T"):
        for i, name in enumerate(("cheap", "knee", "champion")):
            rec = D["picks"][t][i]
            params, cfg = decode_pick(rec)
            a = analyze(params)
            key = f"{t}_{name}"
            out[key] = a
            print(f"\n=== {key}  ({rec['group']}, depth {a['depth']}, "
                  f"design {cfg.get('genotype')}) ===")
            print(f" leaves: {[(l['idx'], l['pnr']) for l in a['leaves']]}")
            print(f" tied leaf squeezing [dB]: {np.round(a['leaf_r_db'],2)}  "
                  f"disp {np.round(np.abs([complex(*d) for d in a['leaf_disp']]),3)}")
            for e in a["events"]:
                print(f" mix node {e['node']:2d} (layer {e['layer']}): "
                      f"{e['left']} + {e['right']}  theta={e['theta_deg']:6.1f} deg "
                      f"(T={e['T']:.3f})  hom x0={e['hom_x0']:+.3f} win {e['window']}")
            fg = a["final_gauss"]
            print(f" final Gaussian: {fg if isinstance(fg, list) else {k: np.round(v,3) for k,v in fg.items()}}")
    with open(os.path.join(REPO, "tree_structure_analysis.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote tree_structure_analysis.json")

if __name__ == "__main__":
    main()

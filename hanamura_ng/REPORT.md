# Hanamura optimization over the artifact-free Pareto fronts

_states processed: 2 | skipped: 0_

Each state has a full before/after architecture in `<group>/<run>_cell<idx>.json` (photon detections, per-mode squeezing dB, passive interferometer U, displacements, success probability) and a before|after Wigner panel `*_wigner.png`. `hanamura_summary.csv` is the flat table.

## Best probability-gain per group

| group | design | ⟨O⟩ | Nc before→after | P before→after | gain | sqdB before→after | negvol b→a |
|---|---|---|---|---|---|---|---|
| B30F_c30_a1p00_b1p00 | B30F | 0.622 | 12→4 | 2.8e-03→3.2e-02 | x13.49 | 15.5→11.2 | 3.422395965465617→2.786042550442338 |
| B30F_c30_a1p41_b1p41 | B30F | 0.667 | 20→8 | 3.9e-04→9.2e-03 | x17.84 | 14.0→13.9 | 1.9503868667542852→1.9818959598841301 |
| B30_c30_a1p00_b1p00 | B30 | 0.554 | 12→4 | 2.4e-03→3.2e-02 | x15.42 | 13.9→11.3 | 3.294261828462922→2.795251190576122 |
| B30_c30_a1p41_b1p41 | B30 | 0.503 | 18→6 | 6.4e-09→4.1e-03 | x223664.62 | 16.1→12.1 | 2.527208758987539→1.7590644750596192 |
| B30_c30_a2p73_b1p41 | B30 | 0.491 | 17→9 | 5.7e-06→3.5e-04 | x97.53 | 16.7→13.0 | 3.1404475586922542→2.291644593299991 |

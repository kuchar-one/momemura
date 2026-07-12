# Hanamura optimization over the artifact-free Pareto fronts

_states processed: 122 | skipped: 21_

Each state has a full before/after architecture in `<group>/<run>_cell<idx>.json` (photon detections, per-mode squeezing dB, passive interferometer U, displacements, success probability) and a before|after Wigner panel `*_wigner.png`. `hanamura_summary.csv` is the flat table.

## Best probability-gain per group

| group | design | ⟨O⟩ | Nc before→after | P before→after | gain | sqdB before→after | negvol b→a |
|---|---|---|---|---|---|---|---|
| 00B_c30_a1p00_b1p00 | 00B | 0.689 | 10→4 | 5.6e-05→1.2e-02 | x40.81 | 9.9→9.7 | 2.711014574281089→2.717687288265934 |
| 00B_c30_a1p41_b1p41 | 00B | 0.887 | 8→4 | 1.5e-06→1.5e-02 | x510.04 | 6.2→10.9 | 2.9739646659132477→3.137141507084507 |
| 00B_c30_a2p73_b1p41 | 00B | 0.573 | 11→7 | 2.4e-06→6.1e-04 | x24.62 | 9.4→13.4 | 3.6654387661022434→1.9455716134813548 |
| 0_c30_a1p00_b1p41 | 0 | 1.037 | 4→2 | 1.3e-03→1.4e-01 | x82.16 | 9.8→8.1 | 1.42411002581453→1.6974057809498255 |
| B30B_c30_a1p00_b1p00 | B30B | 0.557 | 10→6 | 2.2e-07→3.8e-03 | x292.35 | 8.9→14.3 | 2.9345282006265294→2.587793087860766 |
| B30B_c30_a1p41_b1p41 | B30B | 1.005 | 6→2 | 2.8e-04→9.7e-02 | x122.25 | 6.6→7.3 | 3.649577402036135→2.070026911432895 |
| B30_c30_a1p00_b1p41 | B30 | 0.745 | 13→9 | 1.0e-15→2.7e-05 | x23.96 | 5.1→12.0 | 3.874513324087568→2.5902237513328292 |
| B30_c30_a1p41_b1p41 | B30 | 0.694 | 13→9 | 3.9e-16→1.8e-05 | x13.91 | 4.9→11.8 | 3.746866786250026→2.5102347671762546 |

# Why the Hanamura-reduced states look "ruined" (2026-07-11)

Symptom (from `run_hanamura_all.py --limit 3`): after the two-step Hanamura
optimization, the recomputed witness blows up far **past the Gaussian limit** and
fidelity collapses, worse as the reduction factor grows:

```
B30F .../10750  rf2: O 0.6432->1.1033  Nc 20->12  fid 0.689
                rf3: O 0.6432->1.1271  Nc 20->10  fid 0.595
                rf4: O 0.6432->1.7298  Nc 20-> 8  fid 0.340
B30F .../6700   rf2: O 0.5768->0.8977  Nc 20->12  fid 0.678
                rf3: O 0.5768->1.6029  Nc 20-> 8  fid 0.181
                rf4: O 0.5768->2.0102  Nc 20-> 4  fid 0.092
```

⟨O⟩ going **above** the Gaussian limit (G=0.667 for `+`) is the tell: a genuine
sub-Gaussian resource, viewed *in its own frame*, cannot score worse than a
Gaussian. Something is being measured in the wrong frame.

## Root cause 1 (the bug): the final Gaussian correction is never applied

The Hanamura framework preserves the heralded output **only up to a Gaussian
unitary** — this is the whole content of the core-state statement
|ψ⟩ ∝_G (â†+s₀â+δ₀)ⁿ|0⟩ (paper Eqs. 1, 38; thesis `eq:hanamura-particle`). Both
steps inherit it:

- **Step 1 (photon reduction)** "preserv[es] the heralded output **up to a
  Gaussian unitary**" (thesis 4chapter.tex l.315).
- **Step 2 (damping)** "preserves the output **up to a Gaussian unitary**" (paper
  Thm. 3; thesis l.309).

The thesis's own Algorithm 1 therefore *defines* the check as

> verify **F(|ψ_n⟩, Ĝ|ψ_{n'}⟩) ≈ 1 for some Gaussian Ĝ** (l.335)

i.e. the reduced state must be **re-aligned by an optimal single-mode Gaussian
Ĝ** before it is compared to the target.

Our pipeline does not do this. `run_hanamura_all.py` reconstructs the after-state
with `gen_hanamura_data.reduced_full_state`, which heralds the reduced generator
through the **original signal frame**, and then evaluates ⟨O⟩ and fidelity
directly:
- ⟨O⟩ is **not** Gaussian-invariant, so in a stale frame a mis-oriented
  squeeze/rotation inflates it arbitrarily — enough to push it above G even when
  the underlying state is a good resource.
- fidelity is computed as the **raw** overlap |⟨ψ_after|ψ_before⟩|², which
  ignores the Ĝ freedom and so systematically under-reports.

This alone explains the above-Gaussian ⟨O⟩ and much of the fidelity collapse. It
is a **scoring bug**, not a failure of the method. The fix is to score up to a
Gaussian unitary: fidelity via `align_states` (which is exactly the thesis's
`F(|ψ_n⟩, Ĝ|ψ_{n'}⟩)`), and ⟨O⟩_after = min over single-mode Gaussian Ĝ of
⟨Ĝψ_after|O|Ĝψ_after⟩ — the same single-mode-Gaussian freedom the MOME optimizer
already used (via the final-Gaussian gene) to obtain ⟨O⟩_before.

## Root cause 2 (genuine, but smaller once frame-corrected): efficient champions have no photon headroom

Even scored correctly, aggressive reduction genuinely degrades **these** states:

- The reduction is a **local** approximation: it replaces the Fock wavefunction
  φ_n(x) by a rescaled/shifted φ_{n'}(kx−d) that only "matches **near the
  envelope center**" (thesis l.331; paper Fig. 5b, Sec. IV B). Our champions are
  wide, high-amplitude, deeply sub-Gaussian wavefunctions — precisely where the
  center-only match fails globally.
- Our MOME champions were selected by optimizing the **same** non-Gaussian
  quality ⟨O⟩ the reduction would try to preserve, so they already sit near the
  stellar-rank frontier (paper Fig. 2). There are no redundant photons to shed.
  The thesis says as much: "aggressive reductions can raise r_max" and the cost
  is "reshaped rather than uniformly reduced" (l.346–348).

So the monotone fidelity decay with factor (0.69→0.60→0.34) is a real signal, but
its *magnitude* here is exaggerated by root cause 1. Hanamura's factor-of-three
reduction works on **inefficient** generators (redundant photons, poor s₀ for
their n); it cannot improve states that are already efficient.

## Not the cause: normalization / control parameters

The control-parameter formulas in the code match the (corrected) thesis
Algorithm 1 exactly:
- `s0 = (c-d)/(cd-1)` — code `control_parameters` l.112 = thesis l.329.
- `δ0 = (2/√(cd-1))(√((d+1)/(c+1)) β̄x − i√((c+1)/(d+1)) β̄p)` — code l.113–115 =
  thesis l.330.
- `block_from_params` is the exact inverse (round-trip verified algebraically).

The √2 normalization a previous pass fixed lives in the thesis's convention
statement (`C=2Σ_c, β=√2 μ_c`, l.327): it converts Chap. 1 moments into the
vacuum-covariance-=-I convention. The code operates **natively** in that
convention (its cov has vacuum = I; `control_parameters` treats c=d=1 as
vacuum), so no extra factor is needed. Decisively, the mean/covariance convention
is already validated: the **before**-state built from the same `eq["cov"],
eq["mu"]` matches the independent numpy path to 1e-14 and reproduces the L=200
⟨O⟩ (see `NG_VALIDATION_REPORT.md` §4). A √2 error in the mean would have broken
that. So the reduction math is faithful; the problem is purely the missing Ĝ at
scoring time (root cause 1) plus the intrinsic approximation limit (root cause 2).

## Consequence for the earlier 84-state Hanamura front

`plot_ng_fronts.py` states "⟨O⟩ is unchanged by construction (the heralded core
state is preserved)". That is true for the **damping** step but **false** for the
**photon-reduction** step, and that run never recomputed ⟨O⟩_after. Its
"photons halved at preserved ⟨O⟩" claim is therefore unsupported for the
reduction part; the preserved after-negativity (~2.5) does not rescue it, because
Wigner negativity is not target fidelity. Recomputing ⟨O⟩_after (which the new
pipeline does) is what exposed this.

## Fixes (implemented / recommended)

1. **Score the after-state up to a Gaussian unitary** (implemented in
   `run_hanamura_all.py`): report `exp_after_G` = min over single-mode Gaussian Ĝ
   of ⟨O⟩(Ĝψ), and `fid_G` = `align_states` fidelity (thesis F). Keep the raw
   stale-frame numbers alongside, so the size of the frame effect is visible.
2. **Separate the two Hanamura tools** in the front build:
   - **Damping only (reduction factor = 1):** exactly output-preserving (⟨O⟩
     unchanged, fid ≈ 1), only P changes. This is the correct basis for the
     "dominated-before, promoted-after" front — states get promoted purely by the
     probability boost (thesis Table: up to ×155), with **no** quality loss.
   - **Photon reduction (factors ≥ 2):** a genuine quality-vs-(photons,squeezing)
     tradeoff, now scored with Ĝ. Expect real degradation on these efficient
     champions; that is a legitimate negative result, not a bug.
3. The sweep now includes factor 1.0 by default.

## One-line takeaway

The reduction is fine; our **assessment** wasn't: the Hanamura output is only
defined up to a Gaussian unitary, and we scored it without applying that unitary.
Fixing the frame (and using damping-only for the promotion front) is the right
path; aggressive photon reduction still cannot help states that are already
photon-efficient.

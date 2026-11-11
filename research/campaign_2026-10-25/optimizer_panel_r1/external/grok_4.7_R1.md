Muon wins here by forcing full-rank kicks on the update gate, not by turning up the recurrent learning rate. The next experiment should separate that geometry from clipping and from a stale Adam second moment, then attack gate saturation directly.

## A. Why Muon wins

**Established.** For white-noise gradients, bias-corrected Adam has
\[
\mathrm{RMS}\bigl(|\hat m|/\sqrt{\hat v}\bigr)=\sqrt{(1-\beta_1)/(1+\beta_1)}.
\]
At \(\beta_1=0.9\) that is \(0.229\). Fact 4 (median \(0.07\)–\(0.15\), q95 \(0.25\)–\(0.42\)) sits on this floor, so a batch gives no usable per-coordinate signal in any family. Cautious AdamW failing is the same fact: there is no sign agreement to keep. Global \(\ell_2\) clip at \(1.0\) is on for \(80\)–\(94\%\) of steps with mean factor \(0.56\)–\(0.69\), and the head median \(1.30\)–\(1.79\) dwarfs \(W_{hh}\) at \(0.04\)–\(0.11\), so one scalar \(c_t=\min(1,1/\|g\|)\) multiplies every family.

Muon replaces the matrix step by a Newton–Schulz polar factor. With \(M=U\Sigma V^\top\),
\[
\mathrm{polar}(M)=UV^\top,\qquad \sigma_i(\mathrm{polar}(M))=1,
\]
then the step is \(\eta\cdot 0.2\sqrt{\max(m,n)}\cdot\mathrm{polar}(M)\). Entrywise RMS of that factor, after the \(0.2\sqrt{\max(m,n)}\) scale, is \(0.2\). At \(\eta=10^{-3}\) the Muon step RMS is \(2\cdot 10^{-4}\). Adam’s median step is \(\eta\cdot 0.1\approx 10^{-4}\), and its q95 is \(\approx 4\cdot 10^{-4}\). Muon is about \(2\times\) Adam in RMS, not \(10\times\). AdEMAMix, which did inflate steps until the clip factor hit \(0.35\), lost \(0.004\). **Effective-LR increase is not the mechanism.**

Clipping is secondary for the same reason. Undoing a \(0.6\) clip is a \(1.6\times\) rescaling of \(W_{hh}\). That cannot turn a stable rank of \(2\) into \(35\), or a top singular value of \(29\) into \(3\). Newton–Schulz is invariant to a *constant* scale of its argument, so Muon throws away \(c_t\) after momentum mixes steps; time-varying \(c_t\) still reweights history inside \(M\). Real, but too small to explain fact 2.

**Conjecture, tied to facts 2–4 and 6.** Adam’s update gate is a rank-\(2\) object: top singular value \(25\)–\(29\), stable rank \(2.0\)–\(2.4\), against \(5\)–\(12\) and stable rank \(4\)–\(12\) on \(r\) and \(n\). Rough Frobenius check: \(\|W\|_F=\sigma_{\max}\sqrt{\mathrm{sr}}\) gives \(\|W_z\|_F\sim 40\) (Adam) vs \(\sim 19\) (Muon). About \(2\times\) of the singular-value gap is scale; the rest is shape, \(\mathrm{sr}\) \(2\) vs \(35\).

Collapse loop, specific to a saturating gate plus a slow second moment:

1. A direction of \(W_{hh}^{(z)}\) grows. Preactivations leave the linear region of \(\sigma\), so \(\sigma'\to 0\) and the gradient in that direction dies.
2. \(\beta_2=0.999\) keeps \(v\) large for \(\sim 10^3\) steps, which *suppresses* the Adam step that would walk the weight back. Weight decay \(\eta\lambda\sigma\approx 10^{-5}\cdot 29\) is too weak at the top of OneCycle to undo it.
3. Remaining gradient energy moves to whatever subspace is still unsaturated, and the same loop eats it.
4. Terminal state: one or two giant directions, almost no \(W_{hh}\) gradient (fact 3: median \(0.04\)–\(0.11\)), stable rank \(\approx 2\).

Muon has no \(v\). Every step adds a full-rank matrix of fixed RMS, including in directions whose true gradient has vanished, and it refuses to let one singular direction take a larger step than the others. That is an update-side spectral equalizer. It matches the measured rescue (stable ranks \(34\)–\(85\), top singular values \(1.8\)–\(3.2\)) and the \(3\)–\(4\times\) rebound in recurrent gradient norm: \(\sigma'\) is alive again, so BPTT actually returns a signal. Biases ending at RMS \(0.22\) vs \(0.10\) fit the same picture. Under \(\sigma_{\max}\approx 27\) a bias of \(0.1\) cannot move \(\sigma\); under \(\sigma_{\max}\approx 3\) it can.

Fact 6 says why the *update* gate is the one that matters. Pooled error is mostly a per-run level, and labels jump on book events. Holding a level is \(z\approx 1\) between events and a rank-rich jump at events. A rank-\(2\) \(z\) can hold about two modes. The synthetic screen flipping sign (\(-0.005\)) is what you expect if that screen does not have this saturation trap.

### Discriminator

One cold cell, H512×1, 10% data, 3 epochs, two seeds already in the Muon comparison. Four arms, identical data order:

| Arm | Rule | If it ties Muon |
|---|---|---|
| B0 | Current AdamW | — |
| B1 | Current per-gate Muon | — |
| B2 | Momentum, **no polar factor**, RMS matched to Muon: \(M\leftarrow\beta M+G\), \(\Delta=0.2\,M/\|M\|_{\mathrm{rms}}\) | Scale control was enough; equalization was not |
| B3 | AdamW, then project each \(W_{hh}\) gate: \(W\leftarrow W\cdot\min(1,\tau/\|W\|_2)\), \(\tau=(2,3,2)\) for \((r,z,n)\), one power iteration | Weight spectral cap was enough |
| B4 | AdamW with \(\beta_2=0.95\) on GRU blocks only | Stale \(v\), not polarization |

**Prediction:** B1 beats B2 and B4 by most of the \(+0.004\); B3 recovers a large fraction of it and is the one that moves stable rank. Clipping is not in this table on purpose (see C).

Log three series, not just WP: \(\sigma_{\max}(W_{hh}^{(z)})\), stable rank, and saturation \(\mathbb{P}(z\notin[0.05,0.95])\). Also the offset fraction
\[
\phi=1-\frac{\sum_s\sum_{t\in s} w_t(r_t-\bar r_s)^2}{\sum w_t r_t^2}.
\]
A real mechanism hit drops \(\phi\). A WP wiggle with \(\phi\) stuck is not the Muon effect.

**Falsifier for the conjecture:** B2 ties B1, or B1’s \(\phi\) and saturation match B0 while WP still rises. Then the spectra are a side effect of a different trajectory and this section is wrong.

## B. Next steps, in the order I would run them

Seed noise is \(0.0007\)–\(0.0014\) and one Muon interval starts at \(+0.0006\). Pre-register a hit as \(\Delta\mathrm{WP}>0.002\) on **both** cells, same seeds. Anything that needs 15 epochs is the wrong shape: fact 5 already says the metric peaks at epochs 1–3 and then wanders in \(0.658\)–\(0.665\), under a 3-epoch anneal.

### 1. Update-gate bias, AdamW and Muon

Not an optimizer. It is the control fact 2 implies, and it is one line.

\[
b_z \leftarrow b_z + b_0,\qquad b_0\in\{1,2\}
\]
at init only (Gers / Jozefowicz: start the remember-gate open). Everything else fixed.

**Why.** \(\sigma(0)=1/2\) plus the collapse loop above is how you lose the level. A positive \(b_z\) keeps early \(z\) near \(0.7\)–\(0.9\) so BPTT sees the long piecewise-constant labels. Muon already grew biases; AdamW did not.

**Test.** Four fits: AdamW and Muon, \(b_0=0\) (have it) vs \(b_0=1\).

**Falsifier.** Saturation and \(\|g_{W_{hh}}\|\) unchanged and \(|\Delta\mathrm{WP}|<0.001\).

**Expect.** A sizable piece of the Muon gap on AdamW, smaller on Muon. If \(\phi\) drops, this is the lead.

### 2. Spectral cap on \(W_{hh}^{(z)}\) only

After the optimizer step, one power iteration:

\[
s=\|W_{hh}^{(z)}\|_2,\qquad W_{hh}^{(z)}\leftarrow W_{hh}^{(z)}\cdot\min(1,\tau/s),\quad \tau=3.
\]

**Why.** This is arm B3, narrowed to the sick block. Muon’s \(z\)-block already lives at \(3.1\)–\(3.2\); \(\tau=3\) asks whether *parking Adam at that norm* is the whole gain. Established: \(\sigma_{\max}\) of the recurrent matrix bounds the linear map. Conjecture: for a GRU the damage is \(\sigma'\approx 0\), not explosion — fact 3 is vanishing recurrent gradients, not exploding ones.

**Test.** AdamW+\(\tau=3\), Muon+\(\tau=3\). Two to four fits.

**Falsifier.** \(\sigma_{\max}\) sticks at 3 but saturation and \(\phi\) do not move, or they move and WP does not (\(|\Delta|<0.001\)).

**Expect.** AdamW up, Muon flat. If Muon drops, \(3.1\) is a floor it needs, not a ceiling, and the cap is the wrong tool.

### 3. RMS-matched momentum vs polar factor (the real Muon ablation)

On each gate block, clipped exactly as today:

\[
\begin{aligned}
M_t&=\beta M_{t-1}+G_t,\\
N_t&=G_t+\beta M_t &&\text{(Nesterov, your current form)},\\
\Delta^{\mathrm{NS}}_t&=\mathrm{NS}(N_t),\\
\Delta^{\mathrm{rms}}_t&=0.2\cdot N_t\big/\mathrm{RMS}(N_t).
\end{aligned}
\]

Same \(\eta\), same \(0.2\) target RMS, same weight decay. B2 vs B1 in section A.

**Why.** This is the clean split between “fixed step size, Adam’s \(v\) removed” and “singular values of the step equalized.” Fact 4 says removing \(v\) might matter because \(v\) is a noise estimate; fact 2 says equalization is what changed the weights.

**Falsifier.** As in the table: if \(\Delta^{\mathrm{rms}}\) matches \(\Delta^{\mathrm{NS}}\) on WP **and** on stable rank, stop investing in Newton–Schulz variants (joint gates, longer NS, orthogonal manifolds).

**Expect.** \(\Delta^{\mathrm{rms}}\) beats AdamW a little (stale \(v\) was real) and loses most of \(+0.004\) to NS. Stable rank stays low under \(\Delta^{\mathrm{rms}}\).

### 4. Stop clipping Muon blocks; clip Adam families on their own

Keep `clip_grad_norm_` off the concatenated vector. With \(G\) the raw gradient,

\[
\begin{aligned}
G_F&\leftarrow G_F\cdot\min(1, c_F/\|G_F\|),\\
c_{\mathrm{head}}=2,\; c_{\mathrm{in}}=1,\; c_{\mathrm{bias}}=0.5,&\quad c_{W_{ih}}=c_{W_{hh}}=\infty \text{ for Muon blocks}.
\end{aligned}
\]

Muon then uses unclipped \(G\). Its update already has known RMS \(0.2\eta\); a second clip does not change its direction, it only jitters the momentum mix through \(c_t\).

**Why.** Fact 3. This will not reproduce the rank gap (section A), but every later comparison is uninterpretable while the head writes \(c_t\). AdEMAMix already died of this.

**Test.** One AdamW arm and one Muon arm. **Falsifier:** both move by \(<0.001\) and the \(c_t\) histogram was already mild. **Expect** \(+0.001\)–\(0.002\) AdamW, \(\le 0.001\) Muon. Treat a larger AdamW jump as evidence section A underweighted clipping, and rerun B2/B3 under the new clip before believing them.

### 5. Do not extend Muon to the head yet; if you do, isolate it

Head median gradient is already \(1.3\)–\(1.8\). A \(512\times 512\) Muon step at \(0.2\eta\) is the same RMS recipe that, on the recurrent blocks, *rewrote the spectrum*. On the head it is another AdEMAMix-style inflation sitting under a clip the head itself triggers. The output map is \(2\times H\): the polar factor there is two normalized rows, i.e. signed row-normalized SGD, which fact 4 says is also noise.

**Worth one fit only after step 4:** Muon on the first head matrix, AdamW on the \(2\times H\), head clip \(c_{\mathrm{head}}=2\), recurrent Muon unchanged. **Falsifier / expect:** clip factor back near \(0.35\), or WP down \(\sim 0.003\). I would stop if that happens rather than retune \(\eta\) for a week. SOAP/Shampoo on \(H=512\) has the same problem plus a \(512\times 512\) inverse: fact 4 says the second-moment matrix is a noise covariance. No.

### 6. Horizon check, because \(\phi\approx 0.94\) dwarfs \(+0.004\)

If the run level is in the state, TBPTT at 2,000 rows (ten truncations per 20,000-row sequence) throws away the credit for it. One Muon run at truncation 8,000 and one at full sequence, batch cut if memory requires it, **same number of sequences**.

**Falsifier.** \(\phi\) and WP unchanged. Then the offset is not a truncated memory, and no optimizer will touch it: either the features lack a causal baseline, or the head cannot express a per-run level. Confirm with a non-leaking expanding baseline feature before any further optimizer work. **Expect** a gain larger than Muon if the level is identifiable inside the sequence; null if future-price labels are short-horizon and \(\phi\) is just pooled bias.

Joint-gate Muon (one NS on the stacked \(3H\times H\) block) and orthogonal-manifold / Cayley retractions are lower priority. Per-gate NS already equalizes inside \(z\), which is the broken object; stacking gates makes \(r\) and \(n\) compete with \(z\) for the same \(H\) singular directions. Exact orthogonality of \(W_{hh}\) fights the small gain (\(\sigma_{\max}\sim 2\)–\(3\)) Muon actually settled at.

## C. What is misleading, and what to stop

**End-of-anneal Adam statistics.** OneCycle drives \(\eta\to 0\). Fact 4 is the right noise calculation, but it is measured where the step has already been switched off. Recompute \(|m|/\sqrt{v}\) at the epoch-1 peak, when WP is actually moving. A floor there is decisive; a floor only at epoch 3 is the schedule.

**Cross-optimizer spectra of \(W\).** They mix scale, shape, different \(c_t\), and a dead \(\eta\). Report \(\sigma_i/\|W\|_F\), saturation, and \(\phi\), at epoch 1 and at the end. High stable rank from shrinking \(\sigma_{\max}\) can hide a gradient that is still rank-2: check the spectrum of \(G_{W_{hh}^{(z)}}\), not only of \(W\).

**Recurrent grad norms as a score.** The \(3\)–\(4\times\) rise under Muon is what unsaturated \(\sigma'\) *does*. Using it to set learning rates double-counts the mechanism.

**The synthetic CPU screen.** It predicted the wrong sign (\(-0.005\) vs \(+0.004\)). Stop promoting optimizers from it.

**AdEMAMix’s negative result, as stated.** At fixed \(\eta\) the clip went to 99% and factor \(0.35\). That falsifies “AdEMAMix at Adam’s \(\eta\) under global clip,” not slow momentum. Do not rerun it until step 4 exists and \(\eta\) is cut about \(3\times\). Also do not expect it to be the next Muon: coordinate SNR is the wrong place to look, and longer memory of a nonstationary book is a liability (15-epoch runs already oscillate).

**Global clip \(1.0\) as a constant of nature.** It is a head regularizer that accidentally multiplies the GRU. Any new method gets a fake loss or a fake win through \(c_t\).

**Schedules, SWA, EMA, dropout, width past 384.** Fact 5 says they tie or hurt. Width and dropout are done. Three-layer GRU is a small architecture residual, not this search.

**Chasing \(+0.001\).** Below paired seed noise. The Muon cell whose interval starts at \(+0.0006\) is not yet a result you should build on; the \(+0.0043\) \([+0.0020,+0.0072]\) cell is.

**Stopping rule.** If steps 1–3 do not move \(\phi\), stop optimizer shopping. The remaining error is a level the update rule is not identifying. A method that cannot change saturation, stable rank of \(z\), or \(\phi\) will not be another AdamW\(\to\)Muon.

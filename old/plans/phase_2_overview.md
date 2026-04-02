Below is how to think about **Phase 2 ("spike generation + teacher dataset")**: you have a working rate-based ring attractor from Phase 1. Now you need to (a) convert rate signals into spike trains using an inhomogeneous Poisson observation model, (b) bin and optionally smooth those spikes to produce rate estimates at a coarser timescale, and (c) package everything into a reusable, self-describing "teacher dataset" that downstream steps (partial observation, student training, perturbation analysis) can consume without touching the simulator again.

---

## 1) What you are doing (conceptually)

Phase 1 produced a **latent rate signal** \(r_i(t)\): a clean, continuous firing rate for each neuron that exists only inside the model. Real electrophysiology experiments do not observe rates — they observe **discrete spike events**. Phase 2 bridges that gap by:

1. **Generating spikes**: sampling an inhomogeneous Poisson process driven by \(r_i(t)\).
2. **Binning**: aggregating spike events into discrete time bins of width \(dt*\text{bin}\) (which is a \_separate*, typically coarser, timescale than the integration step \(dt\_\text{int}\) from Phase 1).
3. **Smoothing** (optional but standard): convolving binned spike counts with a causal kernel to obtain smooth rate estimates, mimicking what experimentalists do to a PSTH.
4. **Packaging**: saving the spikes, decoded \(\hat{\theta}(t)\), smoothed rates, and all parameter metadata as a self-contained dataset object.

The result is the object your pipeline calls the **teacher dataset** — the "ground truth" you will later subsample (partial observation) and use to train a student RNN.

---

## 2) The math

### A) Inhomogeneous Poisson spike generation

For each neuron \(i\) and integration timestep \(k\) of width \(dt\_\text{int}\), the expected spike count is:

\[
\lambda*i[k] = r_i[k] \cdot dt*\text{int}
\]

The actual spike count is drawn as:

\[
s_i[k] \sim \text{Poisson}(\lambda_i[k])
\]

When \(dt\_\text{int}\) is small and rates are in a reasonable physiological range (e.g., 0–100 Hz), \(\lambda_i[k] \ll 1\) and most draws are 0 or 1, closely approximating a point process.

**Important implementation note:** rates \(r_i[k]\) from Phase 1 must be **non-negative** before passing them to the Poisson sampler. Clamp or verify this explicitly — do not assume the nonlinearity guarantees it under all parameter settings.

### B) Spike binning

After generation, you aggregate spikes into coarser bins of width \(dt*\text{bin} = M \cdot dt*\text{int}\) for some integer \(M \geq 1\):

\[
S*i[m] = \sum*{k=mM}^{(m+1)M - 1} s_i[k]
\]

This gives you an integer-valued array of shape \((T*\text{bin}, N)\), where \(T*\text{bin} = T\_\text{int} / M\).

Typical values: \(dt*\text{int} \approx 0.1\text{ ms}\), \(dt*\text{bin} \approx 10\text{–}25\text{ ms}\), so \(M \approx 100\text{–}250\).

### C) Smoothing (PSTH estimation)

A standard approach is convolution with a **causal boxcar** (rectangular) kernel of width \(w\) bins:

\[
\hat{r}_i[m] = \frac{1}{w} \sum_{j=0}^{w-1} S_i[m - j]
\]

This converts binned counts back into an estimated firing rate (in spikes per bin, or divide by \(dt\_\text{bin}\) to get Hz). A causal kernel is appropriate here because in real experiments you cannot look into the future; if you are doing offline analysis only, a symmetric (acausal) Gaussian kernel is also fine and gives smoother estimates.

---

## 3) The two timescales: \(dt*\text{int}\) vs \(dt*\text{bin}\)

This is the most important conceptual distinction in Phase 2.

| Timescale          | Role                                                       | Typical value |
| ------------------ | ---------------------------------------------------------- | ------------- |
| \(dt\_\text{int}\) | ODE integration step; determines accuracy of rate dynamics | 0.05–0.5 ms   |
| \(dt\_\text{bin}\) | Spike observation bin; what the "student" will see         | 10–25 ms      |

The student RNN in Phase 4 will be trained on binned (and possibly smoothed) data at \(dt*\text{bin}\). It has **no access** to \(dt*\text{int}\) or the underlying rate \(r_i(t)\). You must keep these two timescales strictly separated in your code and dataset metadata.

A common bug: accidentally training the student at \(dt\_\text{int}\) resolution (e.g., because you forget to bin). The sequence length blows up, training is slow, and the learned dynamics are at the wrong timescale.

---

## 4) What you build: concrete code structure

### New classes / modules

**`SpikeGenerator`** (in `src/ring_attractor/spike_generator.py` or similar)

- Takes a rate array \((T\_\text{int}, N)\) and `dt_int`.
- Clamps rates to non-negative.
- Draws Poisson counts: `spikes = rng.poisson(rates * dt_int)`.
- Returns `spikes: (T_int, N)` as `int16` or `int32`.

**`SpikeProcessor`** (same module or separate)

- `bin_spikes(spikes, bin_factor M)` → `(T_bin, N)` integer counts.
- `smooth_spikes(binned, kernel_width w)` → `(T_bin, N)` float rates.
- `dt_bin` property = `dt_int * M`.

**`TeacherDataset`** (in `src/ring_attractor/dataset.py`)

A dataclass or simple container holding:

- `spikes_int: (T_int, N)` — full-resolution spike counts
- `spikes_bin: (T_bin, N)` — binned spike counts
- `rates_smooth: (T_bin, N)` — smoothed rate estimates
- `rates_true: (T_int, N)` — ground-truth rates from Phase 1 (internal use only)
- `theta_hat: (T_int,)` — decoded bump location at integration resolution
- `theta_hat_bin: (T_bin,)` — decoded bump location downsampled to bin resolution
- `params: dict` — all metadata (see Section 6)

**Saving / loading**: use `numpy.savez_compressed` for arrays + `json` for params, or `h5py` if you prefer a single file. Avoid pickle for long-term reproducibility.

### Simulation loop adjustment

Phase 1 ran the simulator and discarded intermediate states (or stored them in a list). Phase 2 requires storing the full rate trajectory. Concretely:

```python
rates_all = np.zeros((T_int, N), dtype=np.float32)
for k in range(T_int):
    rates_all[k] = simulator.perform_single_step(external_input[k])

spikes_int = spike_generator.generate(rates_all)
spikes_bin = spike_processor.bin_spikes(spikes_int, M)
rates_smooth = spike_processor.smooth_spikes(spikes_bin, w)
```

Memory note: for \(N = 100\), \(T\_\text{int} = 10^6\) steps, `float32` rates take ~400 MB. If memory is tight, generate spikes on-the-fly and discard rates after binning. Keep the `rates_true` array only if you explicitly need it for later comparison.

---

## 5) Parameters: what they mean and what to control

### A) `dt_bin` (bin width)

- Sets the temporal resolution the student will see.
- Too small: binned counts are mostly 0/1, noisy, hard to train on.
- Too large: temporal structure is lost; bump movements are smeared.
- Good starting point: 10–20 ms for a ring attractor with \(\tau \approx 10\) ms.

### B) Bin factor `M`

- `M = dt_bin / dt_int`. Must be an integer. If your `T_int` is not divisible by `M`, truncate before binning.

### C) Smoothing kernel width `w`

- `w` bins. A typical choice: 2–5 bins (= 20–100 ms with a 10 ms bin).
- Wider: smoother but introduces more temporal lag and blurs fast transitions.
- Narrower: noisier but more faithful to true timing.
- For sanity checks, compare smoothed rates to ground-truth rates from Phase 1 — they should match up to noise.

### D) Poisson rate scale

- The Poisson generator is driven by `r_i[k]` directly. If Phase 1 rates are dimensionless (not in Hz), you need a scaling factor \(\alpha\) such that \(\lambda*i[k] = \alpha \cdot r_i[k] \cdot dt*\text{int}\).
- A reasonable physiological check: peak firing rates in the bump should be ~30–80 Hz.
- If \(r_i\) from Phase 1 has arbitrary scale, normalize or introduce an explicit gain parameter.

### E) RNG seed

- Phase 2 adds a **new source of randomness** (Poisson draws) on top of Phase 1 (rate noise).
- Use separate RNG objects for the rate simulator and the spike generator, each with its own seed. Store both seeds in the metadata.

---

## 6) Artifacts to produce (non-negotiable)

At the end of Phase 2 you must have:

1. **Full-resolution spike array** `spikes_int: (T_int, N)` — integer counts.
2. **Binned spike array** `spikes_bin: (T_bin, N)` — ready for student consumption.
3. **Smoothed rate estimates** `rates_smooth: (T_bin, N)` — for training and visualization.
4. **Decoded angle** `theta_hat_bin: (T_bin,)` — ground truth label at bin resolution.
5. **Metadata dict** containing at minimum:
   - `N, dt_int, tau, sigma, j0, j1, nonlinearity`
   - `dt_bin, M, kernel_width_bins`
   - `seed_rate, seed_spikes`
   - `T_int, T_bin`
   - `cue_schedule` (if you applied external inputs)
6. **Sanity plots** (see Section 7).

This is the single object passed to every downstream step. If it is missing any of the above, you will find yourself re-running Phase 2 during Phase 4 debugging — a painful outcome.

---

## 7) Sanity checks / plots you must run

Do not proceed to Phase 3 without passing all of these.

### i) Raster plot

Plot a subset of neurons (e.g., 20–40) with spike times as dots, sorted by preferred angle \(\theta_i\). You should see a **traveling diagonal stripe** when the bump moves, or a **stationary stripe** during maintenance. If the raster looks like uniform noise or has no structure, something is wrong with the rates from Phase 1 or the Poisson scaling.

### ii) Firing rate distribution

Histogram of mean firing rates across neurons. Should be bimodal or at least have a clear tail: neurons near the bump peak fire much more than those far away. Neurons far from the bump should be near-silent.

### iii) Smoothed rates vs ground-truth rates

For a few neurons near the bump peak, overlay the smoothed binned rate \(\hat{r}\_i(t)\) (from the Poisson spikes) against the true rate \(r_i(t)\) from Phase 1. They should track each other closely — the smoothed version will be noisier, but the temporal profile should match. This validates your binning and smoothing pipeline.

### iv) Decoded \(\hat{\theta}(t)\) at bin resolution

Downsample the Phase 1 decoded angle to \(dt\_\text{bin}\) resolution and overlay it with a decoded angle computed from the **smoothed rates** (not the true rates). They should agree closely. This validates that the Poisson sampling + smoothing has not destroyed the encoded information.

### v) Total spike count sanity

Verify: `spikes_int.sum(axis=0) / (T_int * dt_int)` ≈ mean firing rate per neuron in Hz. Check this is physiologically plausible (a few Hz background, 30–80 Hz at peak).

---

## 8) Common failure modes

| Symptom                                                           | Likely cause                                                                                        |
| ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Rates passed to Poisson sampler go negative → error or NaN spikes | Nonlinearity did not guarantee non-negativity; clamp explicitly                                     |
| Smoothed rates look flat / no structure                           | Bin width too large, or smoothing kernel too wide                                                   |
| Raster has no diagonal structure                                  | Phase 1 bump is not moving; check cue schedule or rate dynamics                                     |
| `theta_hat_bin` is noisy / jumps erratically                      | Firing rates too low → population vector is unreliable; increase gain or reduce noise               |
| Dataset file is huge                                              | Storing `float64` instead of `float32`; or not compressing; or storing `spikes_int` when not needed |
| Student (Phase 4) learns at wrong timescale                       | Forgot to bin; student is seeing `spikes_int` at `dt_int` resolution                                |

---

## 9) Where Phase 2 fits in the full pipeline

```
Phase 1: Rate ring attractor
    ↓  r_i(t)
Phase 2: Poisson spiking + binning + smoothing
    ↓  TeacherDataset (spikes_bin, rates_smooth, theta_hat_bin, metadata)
Phase 3: Partial observation (drop neurons)
    ↓  Subset of TeacherDataset
Phase 4: Train student RNN
    ↓  Learned dynamics
Phase 5: Perturbation + validation (does the student have a real ring?)
```

Phase 2 is the only place in the pipeline where randomness is introduced via the Poisson observation model. All downstream phases consume the saved dataset and do not re-run the simulator. This is intentional: it makes Phase 3–5 reproducible without re-running Phase 2.

---

## 10) Strong recommendation for your next action

Implement Phase 2 in this strict order:

1. **Add Poisson spike generation** for a single trial, using rates already produced by Phase 1. Produce a raster plot immediately. Verify there is structure.
2. **Add binning** at a chosen `dt_bin`. Verify `spikes_bin` has the right shape and is not uniformly sparse.
3. **Add smoothing** and overlay smoothed rates against Phase 1 rates (sanity check iii above). If they don't match, stop and debug before proceeding.
4. **Build the `TeacherDataset` container** and saving/loading infrastructure. Save one complete dataset with full metadata.
5. **Run all sanity checks** from Section 7 on the saved-then-reloaded dataset to confirm the I/O is lossless and the metadata is complete.

If you skip step 5, you will discover missing metadata only when you need it in Phase 4 — at which point you may not remember what parameters you used.

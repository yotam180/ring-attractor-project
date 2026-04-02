Okay, this is a really important conceptual question. Let me think through this carefully with full context from your pipeline.

## Your confusion is legitimate — but here's the resolution

You're right that the simulation pipeline and the real data have a structural mismatch. But the real data is **not used the same way as the simulated teacher**. Here's exactly what each phase does:

### In the simulation (Phases 1–4)

You need a cue input because you're _constructing_ a ring attractor from scratch and need to plant the bump somewhere. The cue is a **pedagogical tool** — it initializes θ and lets you drive the bump around the ring controllably. The pipeline is:

```
Ring attractor simulator (with cue inputs)
    → spike trains (N neurons, T timesteps)
    → subsample to k neurons / truncate to T' time
    → train student RNN
    → analyze: does student have ring dynamics?
```

### In the real data (Phase 5)

You have NO teacher simulator, NO cue input, and a DIFFERENT N than your teacher. But you don't need any of that. The pipeline is:

```
Real HD recordings (spike trains, N HD cells, T_session)
    → subsample to k neurons / truncate to T' time
    → train autonomous RNN to reproduce firing rates
    → analyze: does student have ring dynamics?
```

The key insight is: **you're not training the RNN to follow cues. You're training an autonomous RNN to reproduce a time series of firing rates.** The RNN gets no input — just learns the recurrent dynamics. If the underlying biology is a ring attractor, a correctly trained RNN on enough data should learn ring attractor dynamics.

The animal's natural head movement is what drives the HD bump around the ring continuously — the vestibular inputs, visual flow, etc. are all doing the work that your simulated cue input was doing. You don't have to replicate that in the RNN; you just train on the resulting activity trace.

### The θ(t) angle from tracking is _not_ an input — it's a label

```
θ(t) is used for:           θ(t) is NOT used for:
- Computing tuning curves   - Training the student RNN
- Verifying cells are HD    - As an input channel
- Post-hoc validation of
  whether the RNN decoded
  angle matches behavior
```

---

## So what does the real data actually contribute to the project?

The simulation (Phases 1–4) establishes the threshold curve in **(k/N, T) space**: below some boundary, student RNNs start getting the mechanism wrong even though their predictive fit is good.

The real data (Phase 5) does three things the simulation cannot:

**1. Validates the threshold curve in biology**  
Your simulation will produce a curve like "you need at least 40% of neurons and 15 min of data to recover a ring." Phase 5 asks: does that same threshold hold for real HD neurons? If it does, the simulation is a good model of the biology.

**2. Gives you "calibrated confidence" for a specific session**  
You take, say, session A3701 (71 HD cells, 42-minute recording). You train a student RNN on all 71 cells for the full 42 min — it should recover ring dynamics. Then you ask: at what subsampling level does ring recovery fail? Your answer becomes: "given this recording, we're confident in the ring conclusion at full observation, but would have been misled if we'd only recorded 30% of neurons."

**3. The 31 sessions give you natural variation for free**  
Sessions have 21–117 HD cells naturally. That IS your "neuron dropout axis" — you're not artificially subsampling a 100-neuron teacher. You can ask whether sessions with fewer HD cells show different RNN dynamics, without any artificial intervention.

---

## The honest picture of what the project shows

The project has two legs:

| Leg                         | What it uses                                    | What it shows                                                                                  |
| --------------------------- | ----------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **Simulation** (Phases 1–4) | Teacher ring attractor + artificial subsampling | Precisely WHERE the threshold is and WHY — causal, controlled                                  |
| **Real data** (Phase 5)     | Biological HD recordings                        | That the same threshold phenomenon occurs in real data, giving the result biological relevance |

The simulation alone would be a pure theory paper. The real data alone would just be "we trained an RNN on HD cells" — not very novel. Together they make the argument: _partial observation causes mechanistic misidentification in a controlled setting (simulation), and here's evidence the same issue affects real recordings (biology)._

---

The real data application is genuinely simpler to implement than the simulation pipeline — no cue inputs, no Poisson spike generation, no teacher–student relationship. You just load NWB files, extract HD cell firing rates during `wake_square`, and run the same student-RNN + fixed-point analysis you already built for Phase 4.

# Q&A

Good, now I have the full picture. Let me answer all four questions clearly.

---

## Q1: How do we actually analyze the student RNN?

Both — you run it AND look at structure. Here's the precise sequence:

**Training phase:** The RNN is trained to reproduce a trajectory of firing rates. It sees the observed firing rate timeseries and learns the recurrent weights that best explain that data (more on this in Q2).

**Evaluation phase — you run it from many initial conditions:**

1. Take the trained RNN (just the weights, autonomous — no input)
2. Initialize it from many random starting states
3. Let it run forward in time and see where it converges
4. Collect all the "landing points"

If it's a **ring attractor**: the landing points form a circle in activity space — a continuous manifold. Infinitely many stable states.

If it's a **discrete attractor**: the landing points form a few isolated blobs — maybe 4, 8, 16 discrete stable positions. The "ring" has been quantized.

The cue is only used during **teacher data generation** to drive the bump around and produce diverse activity. The student never sees cues — it's always evaluated as a fully autonomous system.

---

## Q2: How is the RNN trained? What's the loss?

The student RNN has the same form as your ring attractor simulator:

```
τ dh/dt = -h + φ(J · h)
```

But J (the weights) starts random and is what you're solving for. Training is:

- **Input at each timestep:** the observed binned firing rates `r_obs[t]` — shape `(N_obs,)` — fed in as the initial state or as teacher-forced input
- **Target at each timestep:** `r_obs[t+1]` — the next observed rates
- **Loss:** MSE between what the RNN predicted and what actually happened

```python
# Conceptually:
for t in range(T):
    h[t+1] = rnn_step(h[t])          # autonomous step
    loss += MSE(h[t+1], r_obs[t+1])  # predict next observed state
```

This is called **teacher forcing with BPTT** (backpropagation through time). The RNN is learning: "given the neural population state now, what is it one step later?" If the data comes from a ring attractor, the only recurrent dynamics that can explain arbitrary ring-following trajectories IS a ring attractor — so that's what J learns to encode.

You're not training the RNN to classify or to predict θ. You're training it to be a dynamical system that matches the observed firing rate trajectories. θ is only used post-hoc to verify that the RNN is actually tracking head direction.

---

## Q3: What does "correctly trained RNN learns ring attractor dynamics" mean?

A ring attractor produces activity that lives on a 1D circle manifold. If you give an RNN that data and train it to reproduce it, the RNN is forced to learn recurrent weights that support motion along that manifold. There's no simpler explanation — the data literally traces a circle in activity space, so the RNN must carve out a circular set of stable states in its weight geometry.

When you have **enough data and enough neurons**, the RNN has enough information to correctly reconstruct that manifold. When you have **too few neurons or too short a window**, the RNN sees ambiguous data — maybe it looks like 8 discrete clusters rather than a smooth circle — and it learns a discrete approximation instead. Same trajectory on the surface, wrong underlying geometry.

---

## Q4: Yes — two separate experiments, same machinery

You've understood this exactly correctly. The picture is:

```
Simulated path:
  Ring attractor simulator
    → SpikeGeneratorSimulator → SpikeProcessor
    → rates_smooth (T_bin × N)
    → NeuronDropout(keep_fraction=p) + time truncation
    → Student RNN training + fixed-point analysis
    → ring_score[p, T]   ← a 2D matrix

Biological path:
  NWB file (HD cells)
    → [NEW: NWBLoader → bin spikes → smooth]
    → rates_smooth (T_bin × N_HD)    ← same format
    → NeuronDropout(keep_fraction=p) + time truncation
    → Student RNN training + fixed-point analysis  ← same code
    → ring_score[p, T]   ← same 2D matrix structure
```

The only new code you need for Phase 5 is the NWB loader + binner that produces `rates_smooth` in the same `(T_bin × N)` format your pipeline already expects. After that, `NeuronDropout`, the student RNN, and the ring-score evaluation are **completely reused**.

Note one subtlety in `reduction.py`: `NeuronDropout.reduce()` currently **zeros out** dropped neurons rather than removing them (keeps the array shape at N). For the biological data you probably want to actually **select** the k-neuron subset (slice columns), since N varies across sessions anyway and the student RNN architecture changes with the input dimension. Worth deciding which convention you want before building the bio loader.

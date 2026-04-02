# Task Breakdown: Rate-Only Ring Attractor

This breaks down Phase 1 Step 1 into implementable tasks. Complete these in order before adding Poisson spiking.

---

## Task 1: Project Setup

- [ ] Create `src/` directory structure
- [ ] Create `ring_attractor/` module with `__init__.py`
- [ ] Set up `requirements.txt` with dependencies:
  - numpy
  - matplotlib
  - scipy (optional, for FFT convolution)
- [ ] Create a notebook or script `notebooks/01_rate_ring_demo.ipynb` for interactive development

---

## Task 2: Network Initialization

**Goal**: Set up the basic network state and preferred angles.

### 2.1 Define preferred angles
```python
theta_pref = 2 * np.pi * np.arange(N) / N  # shape (N,)
```

### 2.2 Initialize rate vector
```python
r = np.zeros(N, dtype=np.float32)  # or initialize with small noise
```

### Parameters to expose:
- `N`: number of neurons (start with N=100 or N=200)

### Deliverable:
- Function `initialize_network(N) -> (r, theta_pref)`

---

## Task 3: Connectivity Matrix (Cosine Kernel)

**Goal**: Build the recurrent weight matrix W with ring structure.

### Formula:
```
W_ij = J0 + J1 * cos(theta_i - theta_j)
```

### Implementation:
```python
def build_connectivity(theta_pref, J0, J1):
    N = len(theta_pref)
    # theta_diff[i,j] = theta_pref[i] - theta_pref[j]
    theta_diff = theta_pref[:, None] - theta_pref[None, :]
    W = J0 + J1 * np.cos(theta_diff)
    return W  # shape (N, N)
```

### Parameters to expose:
- `J0`: global inhibition (negative, e.g., -2.0)
- `J1`: structured excitation (positive, e.g., 4.0)

### Sanity check:
- Visualize W as a heatmap (should show banded/circulant structure)
- Each row should be a shifted cosine

### Deliverable:
- Function `build_connectivity(theta_pref, J0, J1) -> W`

---

## Task 4: Nonlinearity Function

**Goal**: Implement activation function φ(x).

### Options:
```python
def relu(x):
    return np.maximum(0, x)

def softplus(x, beta=1.0):
    return np.log1p(np.exp(beta * x)) / beta

def tanh_activation(x, gain=1.0):
    return np.tanh(gain * x)
```

### Recommendation:
- Start with ReLU for simplicity
- Switch to tanh or softplus if activity explodes

### Deliverable:
- Function `phi(x, activation_type='relu', **kwargs) -> x_activated`

---

## Task 5: Euler Integration Step

**Goal**: Implement the core dynamics update.

### Dynamics equation:
```
τ * dr/dt = -r + φ(W @ r + I_ext + σ * ξ)
```

### Euler step:
```
r_new = r + (dt/τ) * (-r + φ(W @ r + I_ext + σ * noise))
```

### Implementation:
```python
def euler_step(r, W, I_ext, dt, tau, sigma, phi_fn, rng):
    noise = rng.standard_normal(r.shape).astype(r.dtype)
    input_total = W @ r + I_ext + sigma * noise
    drdt = -r + phi_fn(input_total)
    r_new = r + (dt / tau) * drdt
    return r_new
```

### Parameters:
- `dt`: integration timestep (e.g., 0.1 ms, must be << τ)
- `tau`: time constant (e.g., 10-20 ms)
- `sigma`: noise level (start with 0, then try small values like 0.1)

### Deliverable:
- Function `euler_step(r, W, I_ext, dt, tau, sigma, phi_fn, rng) -> r_new`

---

## Task 6: Bump Initialization via External Input

**Goal**: Create a localized bump at a target angle.

### Cue input formula:
```
I_cue_i = A * cos(theta_pref_i - theta_target)
```

### Implementation:
```python
def create_cue_input(theta_pref, theta_target, amplitude):
    return amplitude * np.cos(theta_pref - theta_target)
```

### Protocol:
1. Apply cue for T_cue timesteps to "pin" bump to θ_target
2. Remove cue (I_ext = 0) and let network maintain bump

### Deliverable:
- Function `create_cue_input(theta_pref, theta_target, amplitude) -> I_ext`

---

## Task 7: Theta Decoding (Population Vector)

**Goal**: Read out the bump's center angle from activity.

### Formula:
```
z = Σ_i r_i * exp(i * theta_pref_i)
θ_hat = arg(z)
```

### Implementation:
```python
def decode_theta(r, theta_pref):
    # Complex population vector
    z = np.sum(r * np.exp(1j * theta_pref))
    theta_hat = np.angle(z)  # in [-π, π]
    # Optionally wrap to [0, 2π)
    if theta_hat < 0:
        theta_hat += 2 * np.pi
    return theta_hat
```

### Deliverable:
- Function `decode_theta(r, theta_pref) -> theta_hat`

---

## Task 8: Full Simulation Loop

**Goal**: Run complete simulation with cue → maintenance phases.

### Structure:
```python
def simulate_ring_attractor(params, T_cue, T_maintain, theta_target, seed=None):
    rng = np.random.default_rng(seed)
    
    # Initialize
    r, theta_pref = initialize_network(params['N'])
    W = build_connectivity(theta_pref, params['J0'], params['J1'])
    
    n_cue = int(T_cue / params['dt'])
    n_maintain = int(T_maintain / params['dt'])
    n_total = n_cue + n_maintain
    
    # Storage
    rates = np.zeros((n_total, params['N']), dtype=np.float32)
    theta_hat = np.zeros(n_total, dtype=np.float32)
    
    # Simulation
    for k in range(n_total):
        # External input: cue during first phase, zero during maintenance
        if k < n_cue:
            I_ext = create_cue_input(theta_pref, theta_target, params['A_cue'])
        else:
            I_ext = 0.0
        
        # Euler step
        r = euler_step(r, W, I_ext, params['dt'], params['tau'], 
                       params['sigma'], phi_fn, rng)
        
        # Store
        rates[k] = r
        theta_hat[k] = decode_theta(r, theta_pref)
    
    return rates, theta_hat, theta_pref
```

### Deliverable:
- Function `simulate_ring_attractor(params, ...) -> (rates, theta_hat, theta_pref)`

---

## Task 9: Sanity Check Visualizations

**Goal**: Verify the ring attractor works correctly.

### Required plots:
1. **Connectivity heatmap**: W should show circulant/banded structure
2. **Bump profile**: r vs theta_pref at a single timepoint (should be localized)
3. **Activity over time**: heatmap of rates (N × T), bump should be visible as a horizontal band
4. **Decoded θ trajectory**: θ_hat(t), should stay near θ_target (with slow drift if σ > 0)
5. **Bump stability test**: Run without cue after initialization, verify bump persists

### Deliverable:
- Plotting functions in `ring_attractor/visualization.py`
- Demo notebook with all sanity checks passing

---

## Task 10: Parameter Sweep & Verification

**Goal**: Verify behavior across parameter regimes.

### Tests:
1. **Bump existence**: With J1 > threshold, bump forms; below threshold, activity is uniform
2. **Noise-driven diffusion**: With σ > 0, θ_hat should random-walk (compute variance over time)
3. **Input-driven movement**: Apply asymmetric input, bump should move
4. **Stability**: Run long simulation, bump should persist (not collapse or explode)

### Default parameter set (starting point):
```python
params = {
    'N': 100,
    'J0': -2.0,
    'J1': 4.0,
    'tau': 10.0,      # ms
    'dt': 0.5,        # ms (dt/tau = 0.05, safe for Euler)
    'sigma': 0.1,     # noise level
    'A_cue': 1.0,     # cue amplitude
}
```

### Deliverable:
- Verified working parameter set documented in notebook
- Notes on what happens when parameters are out of range

---

## File Structure After Completion

```
project/
├── src/
│   └── ring_attractor/
│       ├── __init__.py
│       ├── network.py          # Tasks 2, 3, 4
│       ├── dynamics.py         # Tasks 5, 6, 7, 8
│       └── visualization.py    # Task 9
├── notebooks/
│   └── 01_rate_ring_demo.ipynb # Tasks 9, 10
├── plans/
│   ├── phase_1_overview.md
│   └── task_breakdown_rate_ring.md (this file)
└── requirements.txt
```

---

## Success Criteria

Before moving to Poisson spiking (next phase), verify:

- [ ] Bump forms when cue is applied
- [ ] Bump persists after cue is removed (maintenance)
- [ ] Decoded θ_hat matches target θ during maintenance (within noise drift)
- [ ] Activity stays bounded (no explosion)
- [ ] θ_hat random-walks slowly when σ > 0 (diffusion along ring manifold)

---

## Next Steps (After This Phase)

Once rate-only ring is working:
1. Add Poisson spiking observation model
2. Create dataset class with spike counts, rates, θ_hat, metadata
3. Implement spike binning and smoothing utilities

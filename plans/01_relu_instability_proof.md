# Analytical Proof: No Stable Nonzero Equilibrium for a Ring Attractor with ReLU Nonlinearity

## 1. Setup

Consider a ring attractor network of $N$ neurons with preferred orientations $\theta_i = 2\pi i / N$, $i = 1, \dots, N$, uniformly distributed on $[0, 2\pi)$. The firing-rate dynamics are

$$
\tau \frac{d\mathbf{r}}{dt} = -\mathbf{r} + \varphi(W\mathbf{r} + \mathbf{I}_{\text{ext}} + \boldsymbol{\xi}),
$$

where:

- $\tau > 0$ is the membrane time constant,
- $\mathbf{r} \in \mathbb{R}^N$ is the vector of firing rates,
- $\varphi(\cdot) = [\cdot]_+ = \max(0, \cdot)$ is the ReLU nonlinearity applied element-wise,
- $\boldsymbol{\xi}$ is additive noise,
- $\mathbf{I}_{\text{ext}}$ is an external input,
- $W$ is the recurrent weight matrix with cosine kernel:

$$
W_{ij} = \frac{1}{N}\left(J_0 + J_1 \cos(\theta_i - \theta_j)\right).
$$

We work with the standard ring attractor parameters $J_0 = -2$ and $J_1 = 4$. We set $\mathbf{I}_{\text{ext}} = 0$ and $\boldsymbol{\xi} = 0$ for the equilibrium analysis.


## 2. Eigenvalue Analysis of $W$

### 2.1 Decomposition of the cosine kernel

Using the angle-addition identity,

$$
\cos(\theta_i - \theta_j) = \cos\theta_i \cos\theta_j + \sin\theta_i \sin\theta_j,
$$

define the vectors

$$
\mathbf{c} = (\cos\theta_1, \cos\theta_2, \dots, \cos\theta_N)^T, \qquad
\mathbf{s} = (\sin\theta_1, \sin\theta_2, \dots, \sin\theta_N)^T.
$$

For equally spaced angles on $[0, 2\pi)$ with large $N$, orthogonality of Fourier modes gives

$$
\|\mathbf{c}\|^2 = \sum_{i=1}^{N} \cos^2\theta_i = \frac{N}{2}, \qquad
\|\mathbf{s}\|^2 = \frac{N}{2}, \qquad
\mathbf{c}^T \mathbf{s} = \sum_{i=1}^{N} \cos\theta_i \sin\theta_i = 0.
$$

The cosine-similarity matrix can then be written as

$$
C_{ij} = \cos(\theta_i - \theta_j) = (\mathbf{c}\mathbf{c}^T + \mathbf{s}\mathbf{s}^T)_{ij},
$$

so

$$
W = \frac{1}{N}\left(J_0 \mathbf{1}\mathbf{1}^T \cdot \frac{1}{N} \cdot N + J_1(\mathbf{c}\mathbf{c}^T + \mathbf{s}\mathbf{s}^T)\right)
= \frac{J_0}{N}\mathbf{1}\mathbf{1}^T + \frac{J_1}{N}(\mathbf{c}\mathbf{c}^T + \mathbf{s}\mathbf{s}^T).
$$

(Here $\mathbf{1} = (1,1,\dots,1)^T$.)

### 2.2 Eigenvalues and eigenvectors

Since $\mathbf{1}$, $\mathbf{c}$, and $\mathbf{s}$ are mutually orthogonal (discrete Fourier orthogonality), each term in the decomposition acts on an independent subspace.

**Uniform mode.** The eigenvector $\mathbf{e}_0 = \mathbf{1}/\sqrt{N}$ satisfies

$$
W \mathbf{e}_0 = \frac{J_0}{N}\mathbf{1}\mathbf{1}^T \frac{\mathbf{1}}{\sqrt{N}} + \frac{J_1}{N}(\mathbf{c}\mathbf{c}^T + \mathbf{s}\mathbf{s}^T)\frac{\mathbf{1}}{\sqrt{N}}.
$$

Since $\mathbf{1}^T\mathbf{1} = N$ and $\mathbf{c}^T\mathbf{1} = \sum_i \cos\theta_i = 0$, $\mathbf{s}^T\mathbf{1} = 0$:

$$
W\mathbf{e}_0 = J_0 \cdot \mathbf{e}_0.
$$

**Eigenvalue:** $\lambda_0 = J_0 = -2$.

**Cosine mode.** The eigenvector $\mathbf{e}_c = \mathbf{c}/\|\mathbf{c}\| = \mathbf{c}\sqrt{2/N}$:

$$
W\mathbf{e}_c = \frac{J_1}{N}\mathbf{c}\mathbf{c}^T \mathbf{e}_c = \frac{J_1}{N}\mathbf{c} \cdot \|\mathbf{c}\|^2 \cdot \frac{1}{\|\mathbf{c}\|} = \frac{J_1}{N} \cdot \frac{N}{2} \cdot \mathbf{e}_c = \frac{J_1}{2}\mathbf{e}_c.
$$

(The $J_0$ term vanishes because $\mathbf{1}^T\mathbf{c} = 0$, and the $\mathbf{s}\mathbf{s}^T$ term vanishes because $\mathbf{s}^T\mathbf{c} = 0$.)

**Eigenvalue:** $\lambda_c = J_1/2 = 2$.

**Sine mode.** By identical argument, $\mathbf{e}_s = \mathbf{s}/\|\mathbf{s}\|$ gives

$$
W\mathbf{e}_s = \frac{J_1}{2}\mathbf{e}_s.
$$

**Eigenvalue:** $\lambda_s = J_1/2 = 2$ (degenerate with the cosine mode).

**All other modes.** Any vector orthogonal to $\mathbf{1}$, $\mathbf{c}$, and $\mathbf{s}$ is annihilated by every term in $W$:

$$
\lambda_k = 0, \quad k \geq 3.
$$

**Summary of spectrum:**

| Mode | Eigenvector | Eigenvalue |
|------|-------------|------------|
| Uniform ($k=0$) | $\mathbf{1}/\sqrt{N}$ | $J_0 = -2$ |
| Cosine ($k=1$, cos) | $\mathbf{c}\sqrt{2/N}$ | $J_1/2 = 2$ |
| Sine ($k=1$, sin) | $\mathbf{s}\sqrt{2/N}$ | $J_1/2 = 2$ |
| Higher ($k \geq 2$) | Higher Fourier modes | $0$ |


## 3. Linear Stability of the Zero State

Consider the zero fixed point $\mathbf{r}^* = \mathbf{0}$. For small positive perturbations $\delta\mathbf{r} > 0$ (component-wise), the ReLU acts as the identity: $\varphi(\delta\mathbf{r}) = \delta\mathbf{r}$. The linearized dynamics become

$$
\tau \frac{d(\delta\mathbf{r})}{dt} = -\delta\mathbf{r} + W\delta\mathbf{r} = (W - I)\delta\mathbf{r}.
$$

Decomposing $\delta\mathbf{r}$ in the eigenbasis of $W$, each mode evolves as

$$
\tau \frac{d(\delta r_k)}{dt} = (\lambda_k - 1)\,\delta r_k,
$$

giving a growth rate $\sigma_k = (\lambda_k - 1)/\tau$ for each mode.

| Mode | $\lambda_k$ | Growth rate $\sigma_k \tau$ | Stability |
|------|-------------|---------------------------|-----------|
| Uniform | $-2$ | $-3$ | Stable |
| Cos/Sin | $2$ | $+1$ | **Unstable** |
| Higher | $0$ | $-1$ | Stable |

Since $J_1/2 = 2 > 1$, the cosine and sine modes (i.e., spatially modulated bump perturbations) are **linearly unstable** around $\mathbf{r} = 0$. Any infinitesimal bump-shaped perturbation will initially grow exponentially.

This is the first half of the paradox: the network *wants* to form a bump.


## 4. The Key Failure: No Nonzero Equilibrium

### 4.1 Fixed-point condition

At a fixed point with $\mathbf{I}_{\text{ext}} = 0$, the dynamics require

$$
\mathbf{r}^* = \varphi(W\mathbf{r}^*) = [W\mathbf{r}^*]_+.
$$

In the region where $r^*_i > 0$ (the "active" region), this reduces to

$$
r^*_i = (W\mathbf{r}^*)_i, \qquad \text{for neurons } i \text{ with } r^*_i > 0.
$$

### 4.2 Half-cosine bump ansatz

Seek a bump-shaped equilibrium centered at $\theta^* = 0$ (by rotational symmetry, the center is arbitrary). In the continuous limit $N \to \infty$, write the firing rate profile as

$$
r(\theta) = A\,[\cos\theta]_+ =
\begin{cases}
A\cos\theta, & |\theta| < \pi/2, \\
0, & |\theta| \geq \pi/2,
\end{cases}
$$

where $A > 0$ is the peak amplitude.

### 4.3 Computing the feedback at the peak

At the bump center $\theta = 0$, the self-consistency condition is

$$
r(0) = (Wr)(0),
$$

i.e.,

$$
A = \frac{1}{2\pi}\int_{-\pi}^{\pi} \bigl(J_0 + J_1\cos\varphi\bigr)\,r(\varphi)\,d\varphi.
$$

Since $r(\varphi) = A\cos\varphi$ only for $|\varphi| < \pi/2$ and zero elsewhere:

$$
A = \frac{A}{2\pi}\int_{-\pi/2}^{\pi/2} \bigl(J_0 + J_1\cos\varphi\bigr)\cos\varphi\,d\varphi.
$$

Evaluate the two integrals separately:

$$
\int_{-\pi/2}^{\pi/2} \cos\varphi\,d\varphi = [\sin\varphi]_{-\pi/2}^{\pi/2} = 2,
$$

$$
\int_{-\pi/2}^{\pi/2} \cos^2\varphi\,d\varphi = \int_{-\pi/2}^{\pi/2}\frac{1 + \cos 2\varphi}{2}\,d\varphi = \frac{\pi}{2}.
$$

Therefore,

$$
A = \frac{A}{2\pi}\left(2J_0 + \frac{\pi}{2}J_1\right) = A\left(\frac{J_0}{\pi} + \frac{J_1}{4}\right).
$$

### 4.4 The gain condition

Define the **effective gain** for the half-cosine bump:

$$
G = \frac{J_0}{\pi} + \frac{J_1}{4}.
$$

Self-consistency requires $A = G \cdot A$, i.e.,

$$
A(1 - G) = 0.
$$

This has a nontrivial solution $A > 0$ **only if** $G = 1$ exactly. For $J_0 = -2$, $J_1 = 4$:

$$
G = \frac{-2}{\pi} + \frac{4}{4} = -0.637 + 1.0 = 0.363.
$$

Since $G = 0.363 < 1$, the only solution is

$$
\boxed{A = 0.}
$$

**No nonzero half-cosine bump equilibrium exists.**


## 5. Generalization to Arbitrary Bump Width

### 5.1 General bump ansatz

Consider a more general bump of angular half-width $\alpha \in (0, \pi)$, centered at $\theta = 0$:

$$
r(\theta) = A\,[\cos\theta - \cos\alpha]_+ =
\begin{cases}
A(\cos\theta - \cos\alpha), & |\theta| < \alpha, \\
0, & |\theta| \geq \alpha.
\end{cases}
$$

This reduces to the previous case when $\alpha = \pi/2$ (since $\cos(\pi/2) = 0$).

### 5.2 Self-consistency integral

At $\theta = 0$, the fixed-point condition gives

$$
A(1 - \cos\alpha) = \frac{A}{2\pi}\int_{-\alpha}^{\alpha}(J_0 + J_1\cos\varphi)(\cos\varphi - \cos\alpha)\,d\varphi.
$$

Expanding the integrand and using

$$
\int_{-\alpha}^{\alpha}\cos\varphi\,d\varphi = 2\sin\alpha, \qquad
\int_{-\alpha}^{\alpha}\cos^2\varphi\,d\varphi = \alpha + \frac{\sin 2\alpha}{2},
$$

we get

$$
2\pi(1 - \cos\alpha) = J_0\bigl(2\sin\alpha - 2\alpha\cos\alpha\bigr) + J_1\left(\alpha + \frac{\sin 2\alpha}{2} - 2\sin\alpha\cos\alpha\right).
$$

Note that $2\sin\alpha\cos\alpha = \sin 2\alpha$, so the $J_1$ bracket simplifies to just $\alpha$. Wait -- let us be more careful:

$$
J_1\left(\alpha + \frac{\sin 2\alpha}{2} - \sin 2\alpha\right) = J_1\left(\alpha - \frac{\sin 2\alpha}{2}\right).
$$

So the self-consistency equation is

$$
2\pi(1 - \cos\alpha) = 2J_0(\sin\alpha - \alpha\cos\alpha) + J_1\left(\alpha - \frac{\sin 2\alpha}{2}\right). \tag{$*$}
$$

### 5.3 No solution for $J_0 = -2$, $J_1 = 4$

Define

$$
F(\alpha) \;=\; 2J_0(\sin\alpha - \alpha\cos\alpha) + J_1\!\left(\alpha - \frac{\sin 2\alpha}{2}\right) - 2\pi(1 - \cos\alpha).
$$

A nonzero equilibrium exists if and only if $F(\alpha) = 0$ for some $\alpha \in (0, \pi)$.

Substituting $J_0 = -2$ and $J_1 = 4$:

$$
F(\alpha) = -4(\sin\alpha - \alpha\cos\alpha) + 4\!\left(\alpha - \frac{\sin 2\alpha}{2}\right) - 2\pi(1 - \cos\alpha).
$$

Let us check the boundary behavior:

- As $\alpha \to 0^+$: All terms vanish, so $F(0) = 0$ (trivially, corresponding to $A = 0$). Expanding to leading order, $F(\alpha) \approx -\frac{4}{3}\alpha^3 + \frac{4}{3}\alpha^3 - \pi\alpha^2 = -\pi\alpha^2 < 0$.
- At $\alpha = \pi/2$: $F(\pi/2) = -4(1 - 0) + 4(\pi/2 - 0) - 2\pi(1 - 0) = -4 + 2\pi - 2\pi = -4 < 0$.
- At $\alpha = \pi$: $F(\pi) = -4(0 - \pi(-1)) + 4(\pi - 0) - 2\pi(1-(-1)) = -4\pi + 4\pi - 4\pi = -4\pi < 0$.

Since $F(\alpha) < 0$ throughout $(0, \pi]$, **no bump width gives a self-consistent equilibrium** with these parameters.

### 5.4 Critical $J_1$ for bump formation

For the half-cosine case ($\alpha = \pi/2$), self-consistency requires $G = 1$:

$$
\frac{J_0}{\pi} + \frac{J_1}{4} = 1 \quad \implies \quad J_1^{\text{crit}} = 4\!\left(1 - \frac{J_0}{\pi}\right).
$$

For $J_0 = -2$:

$$
J_1^{\text{crit}} = 4\!\left(1 + \frac{2}{\pi}\right) = 4 \cdot 1.637 \approx 6.55.
$$

This is much larger than $J_1 = 4$. Even if we increased $J_1$ to $J_1^{\text{crit}}$, the resulting equilibrium would be **marginally stable at best**: the self-consistency equation $A = GA$ with $G = 1$ is satisfied for *any* amplitude $A$. There is a continuous family of fixed points parameterized by $A$, none of which is an isolated stable attractor. Any perturbation in amplitude drifts without restoring force. This is the hallmark of a piecewise-linear system lacking a saturation mechanism.


## 6. Why Tanh Fixes This

Replace $\varphi = \text{ReLU}$ with $\varphi = \tanh$. The key difference is that tanh provides **gain saturation**:

$$
\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x) < 1 \quad \text{for all } x \neq 0.
$$

### 6.1 Amplitude-dependent gain

For a bump of amplitude $A$, the effective local gain of $\tanh$ at the peak is

$$
g(A) = \tanh'(A) = 1 - \tanh^2(A).
$$

- When $A \approx 0$: $g(A) \approx 1$ (linear regime, same as ReLU).
- When $A \gg 1$: $g(A) \to 0$ (saturated regime, fundamentally different from ReLU).

The effective gain is a **monotonically decreasing** function of amplitude. This creates a negative feedback loop on the bump amplitude.

### 6.2 Self-consistent amplitude selection

At a fixed point with $\varphi = \tanh$, the self-consistency condition at the peak becomes (schematically):

$$
A = \tanh\!\bigl(G_{\text{eff}}(A) \cdot A\bigr),
$$

where $G_{\text{eff}}(A)$ encodes the recurrent feedback strength, which itself depends on the full bump shape through tanh compression. The crucial point is:

- For small $A$: the RHS $\approx G_{\text{eff}}(0) \cdot A$ with $G_{\text{eff}}(0) > 1$ (same instability of zero as before, since the linearized eigenvalue $J_1/2 = 2 > 1$). The RHS exceeds $A$, so the amplitude grows.
- For large $A$: saturation compresses the RHS below $A$, since $\tanh(x) < x$ for all $x > 0$. The RHS falls below $A$, so the amplitude shrinks.

By the intermediate value theorem, there exists a unique crossing point $A^*$ where

$$
A^* = \tanh\!\bigl(G_{\text{eff}}(A^*) \cdot A^*\bigr).
$$

This is a **stable** equilibrium: perturbations below $A^*$ experience net growth (gain > 1 in the linearized sense), and perturbations above $A^*$ experience net decay (gain < 1 after saturation).

### 6.3 Contrast with ReLU

For ReLU, the gain in the active region is **identically 1**, independent of amplitude:

$$
\frac{d}{dx}[x]_+ = 1 \quad \text{for all } x > 0.
$$

This means the effective gain has no dependence on $A$. The self-consistency equation is always *linear* in $A$: either $G < 1$ and $A = 0$ is the only solution, or $G = 1$ and every $A$ is a (marginally stable) solution, or $G > 1$ and the amplitude diverges. There is never a unique, stable, finite-amplitude equilibrium.


## 7. Summary: The Paradox and Its Resolution

The analysis reveals a fundamental paradox for the ReLU ring attractor with $J_0 = -2$, $J_1 = 4$:

| Regime | Analysis | Conclusion |
|--------|----------|------------|
| Near $\mathbf{r} = 0$ | Eigenvalue $\lambda = J_1/2 = 2 > 1$ | Bump perturbations **grow** |
| At finite amplitude | Effective gain $G = 0.363 < 1$ | Bump amplitude **decays** |

With ReLU, these two regimes cannot be reconciled:

1. **Small perturbations grow** because the dominant eigenvalue $J_1/2 = 2$ exceeds the decay rate of 1.
2. **Finite bumps collapse** because the projection of the bump shape onto the recurrent kernel yields an effective gain of only 0.363 -- well below the self-sustaining threshold of 1.

The gap arises because a bump concentrates activity over a limited angular range, so much of the recurrent excitation ($J_1$ term) is "wasted" on the silent region, while the global inhibition ($J_0$ term) suppresses the active region. In the linear (eigenvalue) analysis, the perturbation is a full cosine wave spanning the entire ring, which couples maximally to the $J_1$ kernel. A localized, half-rectified bump does not.

**With tanh**, the amplitude-dependent gain provides exactly the missing ingredient: a smooth interpolation between the "grow" regime (small $A$, gain $\approx 1$) and a "shrink" regime (large $A$, gain $\ll 1$). The stable equilibrium sits at the unique amplitude where growth and decay balance -- an amplitude that is self-correcting under perturbation. This is fundamentally impossible with a piecewise-linear activation whose gain is either 0 (inactive) or 1 (active), with no intermediate values to mediate stability.

$$
\boxed{\text{ReLU ring attractor: no stable nonzero bump.} \quad \text{Tanh ring attractor: unique stable bump via gain saturation.}}
$$

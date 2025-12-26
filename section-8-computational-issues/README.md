# Section 8 – Computational and Numerical Issues in HMMs

The previous sections described the **theoretical** and **algorithmic** aspects of HMMs. This section focuses on

- **Numerical stability** (underflow, overflow, log-domain computations);
- **Time and memory complexity** of inference and learning;
- **Approximate inference** when exact methods are too expensive.

Zucchini et al. devote substantial attention to **implementation details** (especially in R code); here we formalize and extend those considerations.

---

## 8.1 Numerical Stability

### 8.1.1 Underflow in the Forward Algorithm

Recall the unnormalized forward variables
\[
\tilde{\alpha}_t(i) = \mathbb{P}(S_t=i, Y_{1:t}=y_{1:t}).
\]

For moderate \(T\), these values can be extremely small:

- If typical emission probabilities are around \(10^{-2}\), then \(\prod_{t=1}^T 10^{-2} = 10^{-2T}\) quickly underflows in double precision.

Therefore, naive implementations of the forward recursion lead to **numerical zeros**, even when the true probability is non-zero.

### 8.1.2 Scaling Strategy

A standard solution (used systematically in Zucchini et al.) is to **renormalize** at each time step.

Define scaling constants
\[
 c_t = \sum_{i=1}^K \tilde{\alpha}_t(i),
\]
and scaled forward variables
\[
 \hat{\alpha}_t(i) = \frac{\tilde{\alpha}_t(i)}{c_t}.
\]

Then
\[
 \sum_i \hat{\alpha}_t(i) = 1, \quad \hat{\alpha}_t(i) = \mathbb{P}(S_t=i \mid Y_{1:t}=y_{1:t}).
\]

Moreover,
\[
 L(\theta; y_{1:T}) = \prod_{t=1}^T c_t,
\]
so the log-likelihood is
\[
 \ell(\theta; y_{1:T}) = \sum_{t=1}^T \log c_t.
\]

This approach keeps all computations in a numerically safe range while preserving the **exact values** of probabilities (up to floating-point rounding).

### 8.1.3 Log-Domain Computations

An alternative is to work entirely in the **log domain**. Let
\[
 a_t(i) = \log \tilde{\alpha}_t(i).
\]

Then the recursion becomes
\[
 a_{t+1}(j) = \log f_j(y_{t+1}) + \log \sum_{i=1}^K e^{a_t(i) + \log \gamma_{ij}}.
\]

To compute \(\log \sum_i e^{z_i}\) stably, use the **log-sum-exp** identity:
\[
 \log \sum_i e^{z_i} = m + \log \sum_i e^{z_i - m}, \quad m = \max_i z_i.
\]

This avoids overflow/underflow as long as \(z_i\) are in representable range. Similar tricks apply in backward, Viterbi, and EM computations.

### 8.1.4 Backward and Viterbi Stability

- **Backward recursion**: Use either scaling synchronized with forward scaling or log-domain operations to avoid accumulation of tiny values.
- **Viterbi algorithm**: Since it already works with **max-products**, it is natural to convert to **max-sum** in log space, which improves stability and interpretability (additive costs).

---

## 8.2 Computational Complexity

### 8.2.1 Inference for a Single Sequence

Let \(K\) be the number of states and \(T\) the sequence length.

- **Forward algorithm:** For each \(t\), computing \(\tilde{\alpha}_{t+1}(j)\) requires a sum over \(i=1,\dots,K\), so the cost per time step is \(\mathcal{O}(K^2)\). Total cost is \(\mathcal{O}(K^2 T)\).
- **Backward algorithm:** Same complexity as forward.
- **Viterbi algorithm:** Also \(\mathcal{O}(K^2 T)\) due to the max over \(i\) for each \(j,t\).

**Memory usage:**

- Forward alone can be done with \(\mathcal{O}(K)\) memory if only the likelihood is needed;
- Forward–backward typically stores \(\mathcal{O}(K T)\) values (e.g. \(\hat{\alpha}_t\), \(\hat{\beta}_t\)) unless one uses streaming or **checkpointing** strategies.

### 8.2.2 EM / Baum–Welch Complexity

Each EM iteration involves:

- A full **forward–backward pass** per sequence: \(\mathcal{O}(K^2 T)\);
- Simple **M-step updates** costing \(\mathcal{O}(K^2 T)\) for transitions and \(\mathcal{O}(K T)\) for emissions.

If there are \(N\) independent sequences of average length \(T\), the per-iteration cost is \(\mathcal{O}(N K^2 T)\).

Zucchini et al. highlight that, for moderate \(K\) (say \(K \le 10\)) and reasonably long time series, EM is typically very fast on modern hardware.

### 8.2.3 Scalability Considerations

For large-scale problems:

- Reducing **state space size** or enforcing **sparsity** in \(\boldsymbol{\Gamma}\) (many zeros) can reduce the \(K^2\) factor;
- Parallelization over sequences is straightforward;
- GPU implementations can exploit the regular structure of matrix–vector products.

---

## 8.3 Approximate Inference Methods

When exact \(\mathcal{O}(K^2 T)\) inference is too costly or when the model is more complex (e.g. continuous-state or nonparametric HMMs), **approximate methods** are used.

### 8.3.1 Truncated and Beam Search for Viterbi

For very large \(K\) or long sequences, one can approximate Viterbi by:

- **Beam search:** At each time step, keep only the top \(B\) partial paths (states) according to their scores; complexity becomes \(\mathcal{O}(B K T)\) with trade-off between accuracy and speed.

### 8.3.2 Particle Filters (Sequential Monte Carlo)

For continuous-state models, **particle filters** approximate filtering distributions by a weighted set of particles \(\{(X_t^{(n)}, w_t^{(n)})\}\). For finite-state HMMs, particle filters are not usually necessary, but similar ideas can be applied to **very large or structured state spaces**.

### 8.3.3 Variational Inference

In complex HMM variants (e.g. nonparametric HMMs, switching SSMs), one often uses **variational approximations**:

- Posit a factorized form for the posterior over states (e.g. mean-field or structured);
- Optimize an **ELBO**, similar in spirit to EM but with additional approximations;
- Retain forward–backward-like updates, but in an approximate model.

### 8.3.4 Online and Streaming Algorithms

For streaming data, one can use:

- **Online EM**: update parameter estimates incrementally using stochastic approximation to the E-step statistics;
- **Recursive maximum likelihood** methods (e.g. gradient ascent with step sizes \(\eta_t\)).

These algorithms rely heavily on **ergodic and mixing properties** discussed in Section 6.

---

## 8.4 Implementation Notes (Zucchini et al.)

Zucchini et al. provide practical guidance on implementing HMMs, including:

- Careful use of **scaling** in forward–backward algorithms;
- Vectorized operations (e.g. in R or MATLAB) to exploit matrix structures;
- Diagnostics for **convergence** and **numerical issues** (e.g. checking that filtering probabilities remain normalized).

These considerations are essential for turning theoretical algorithms into **robust software**.

---

## 8.5 Summary

This section covered the **algorithmic engineering** side of HMMs:

- Handling **underflow and overflow** via scaling and log-domain computations;
- Understanding the **time and space complexity** of inference and EM;
- Employing **approximate methods** when exact inference is infeasible.

These issues are critical in real-world applications, even though the mathematical structure of HMMs remains the same.

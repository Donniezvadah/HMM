# Section 5 – Parameter Estimation in Hidden Markov Models

This section studies **parameter estimation** for finite-state HMMs, focusing on:

- **Maximum likelihood estimation (MLE)** and its properties;
- The **EM / Baum–Welch algorithm**, including derivation and monotonicity;
- **Identifiability theory** and label switching.

We follow the structure of **Zucchini et al.**, Chapters 3–4, and the rigorous development in **Cappé, Moulines, Rydén (2005)**.

Let $\theta = (\boldsymbol{\delta}, \boldsymbol{\Gamma}, \phi_1,\dots,\phi_K)$ collect all parameters (initial distribution, transition matrix, emission parameters). Given data $y_{1:T}$, we aim to estimate $\theta$.

---

## 5.1 Maximum Likelihood Estimation

### 5.1.1 Definition

Given observed data $y_{1:T}$, the **likelihood function** is
$$
L_T(\theta) := L(\theta; y_{1:T}) = \mathbb{P}_\theta(Y_{1:T} = y_{1:T}),
$$
with log-likelihood
$$
\ell_T(\theta) = \log L_T(\theta).
$$

A **maximum likelihood estimator** (MLE) $\hat{\theta}_T$ is any point in
$$
\hat{\theta}_T \in \arg\max_{\theta \in \Theta} \ell_T(\theta).
$$

Because $\Theta$ is constrained (simplices, stochastic matrices), many implementations reparameterize (e.g. via logits) to perform unconstrained optimization.

### 5.1.2 Non-Convexity and Local Maxima

The log-likelihood $\ell_T(\theta)$ for HMMs is typically **non-convex**:

- Hidden states introduce **latent-variable structure**;
- Symmetries (permutations of states) yield **multiple equivalent maxima**;
- There may be **spurious local maxima** unrelated to the true parameter.

Consequences:

- Gradient-based methods can get trapped in local optima;
- EM (below) converges to a **local stationary point**, not necessarily a global maximum;
- Good **initialization** (e.g. k-means clustering on observations, or simpler models) is critical in practice (as emphasized by Zucchini et al.).

### 5.1.3 Label Switching and Equivalence Classes

For any permutation $\sigma$ of $\{1,\dots,K\}$, define a **permuted parameter** $\theta^{\sigma}$ by

- $\delta^{\sigma}_i = \delta_{\sigma^{-1}(i)}$;
- $\gamma^{\sigma}_{ij} = \gamma_{\sigma^{-1}(i), \sigma^{-1}(j)}$;
- Emission parameters re-labeled: $\phi^{\sigma}_i = \phi_{\sigma^{-1}(i)}$.

Then
$$
L_T(\theta^{\sigma}) = L_T(\theta)
$$
for all $T$ and all data sequences. Thus, parameters are at best **identifiable up to permutation** of hidden states.

This **label switching** means:

- The MLE is only unique up to permutation;
- Post-processing (e.g. ordering states by mean of emissions) is often used to select a canonical labeling (as in Zucchini et al.).

---

## 5.2 EM / Baum–Welch Algorithm

### 5.2.1 General EM Framework

Suppose $Y$ is observed data and $S$ is latent/hidden data. The EM algorithm iteratively maximizes the log-likelihood $\ell(\theta) = \log p_\theta(Y)$ via:

1. **E-step:** Compute
   $$
   Q(\theta \mid \theta^{(k)})
   = \mathbb{E}_{\theta^{(k)}}[\log p_\theta(Y,S) \mid Y].
   $$
2. **M-step:** Set
   $$
   \theta^{(k+1)} \in \arg\max_\theta Q(\theta \mid \theta^{(k)}).
   $$

EM guarantees **non-decreasing likelihood**: $\ell(\theta^{(k+1)}) \ge \ell(\theta^{(k)})$.

In HMMs, $S = S_{1:T}$ (hidden states) and $Y = Y_{1:T}$ (observations).

### 5.2.2 Complete-Data Log-Likelihood for HMMs

Recall (Section 3.2) that the **complete-data log-likelihood** is
$$
\log p_\theta(S_{1:T},Y_{1:T})
= \log \delta_{S_1} + \sum_{t=2}^T \log \gamma_{S_{t-1}, S_t}
  + \sum_{t=1}^T \log f_{S_t}(y_t; \phi_{S_t}).
$$

Thus,
$$
\begin{aligned}
Q(\theta \mid \theta^{(k)})
&= \mathbb{E}_{\theta^{(k)}}[\log p_\theta(S_{1:T},Y_{1:T}) \mid Y_{1:T}=y_{1:T}] \\
&= \sum_i \mathbb{E}[\mathbf{1}_{\{S_1=i\}} \mid Y] \log \delta_i \\
&\quad + \sum_{t=2}^T \sum_{i,j} \mathbb{E}[\mathbf{1}_{\{S_{t-1}=i,S_t=j\}} \mid Y] \log \gamma_{ij} \\
&\quad + \sum_{t=1}^T \sum_i \mathbb{E}[\mathbf{1}_{\{S_t=i\}} \mid Y] \log f_i(y_t; \phi_i).
\end{aligned}
$$

Define the **expected sufficient statistics** under $\theta^{(k)}$:
$$
\gamma_t^{(k)}(i) = \mathbb{P}_{\theta^{(k)}}(S_t=i \mid Y_{1:T}),
$$
$$
\xi_t^{(k)}(i,j) = \mathbb{P}_{\theta^{(k)}}(S_{t-1}=i, S_t=j \mid Y_{1:T}).
$$

These are computed using the **forward–backward algorithm** (Section 4.2).

Then
$$
\begin{aligned}
Q(\theta \mid \theta^{(k)})
&= \sum_i \gamma_1^{(k)}(i) \log \delta_i \\
&\quad + \sum_{t=2}^T \sum_{i,j} \xi_t^{(k)}(i,j) \log \gamma_{ij} \\
&\quad + \sum_{t=1}^T \sum_i \gamma_t^{(k)}(i) \log f_i(y_t; \phi_i).
\end{aligned}
$$

### 5.2.3 M-Step Updates

Maximizing $Q$ over $\theta$ subject to the usual constraints yields closed-form updates for $\boldsymbol{\delta}$ and $\boldsymbol{\Gamma}$, and often for $\phi_i$ (for exponential-family emissions).

- **Initial distribution:**
  $$
  \delta_i^{(k+1)} = \gamma_1^{(k)}(i).
  $$
- **Transition probabilities:** for each $i$,
  $$
  \gamma_{ij}^{(k+1)}
  = \frac{\sum_{t=2}^T \xi_t^{(k)}(i,j)}{\sum_{t=2}^T \sum_{j'} \xi_t^{(k)}(i,j')}.
  $$

For **emission parameters** (e.g. Gaussian), the M-step corresponds to a **weighted maximum likelihood** with weights $\gamma_t^{(k)}(i)$. For instance, if $f_i$ is normal $\mathcal{N}(\mu_i,\sigma_i^2)$,
$$
\mu_i^{(k+1)} = \frac{\sum_{t=1}^T \gamma_t^{(k)}(i) y_t}{\sum_{t=1}^T \gamma_t^{(k)}(i)},
$$
$$
(\sigma_i^2)^{(k+1)} = \frac{\sum_{t=1}^T \gamma_t^{(k)}(i) (y_t - \mu_i^{(k+1)})^2}{\sum_{t=1}^T \gamma_t^{(k)}(i)}.
$$

Zucchini et al. work out these updates for many common emission families (Poisson, normal, etc.).

### 5.2.4 EM as Coordinate Ascent on an Evidence Lower Bound

Define a distribution $q(S_{1:T})$ over state sequences. Then
$$
\log p_\theta(Y)
= \mathcal{F}(q,\theta) + \mathrm{KL}\bigl(q(S_{1:T}) \Vert p_\theta(S_{1:T} \mid Y)\bigr),
$$
where the **variational free energy** (or ELBO) is
$$
\mathcal{F}(q,\theta) = \mathbb{E}_q[\log p_\theta(S_{1:T},Y)] + H(q),
$$
with entropy $H(q) = -\mathbb{E}_q[\log q(S_{1:T})]$.

Since KL is non-negative,
$$
\mathcal{F}(q,\theta) \le \log p_\theta(Y),
$$
with equality iff $q = p_\theta(S_{1:T} \mid Y)$.

EM alternates:

- **E-step:** Set $q^{(k)} = p_{\theta^{(k)}}(S_{1:T} \mid Y)$, which maximizes $\mathcal{F}(q, \theta^{(k)})$ over $q$;
- **M-step:** Maximize $\mathcal{F}(q^{(k)}, \theta)$ over $\theta$, which is equivalent to maximizing $Q(\theta \mid \theta^{(k)})$.

Thus EM is **coordinate ascent** on $\mathcal{F}$, and therefore
$$
\ell(\theta^{(k+1)}) \ge \ell(\theta^{(k)}).
$$

### 5.2.5 Convergence Properties

Under mild conditions (continuity of $\ell$, compactness of parameter space or coercivity), the EM sequence $\{\theta^{(k)}\}$:

- Has **non-decreasing likelihood**;
- Every **limit point** is a **stationary point** of the likelihood (satisfies first-order conditions);
- Global convergence to the **global maximum** is not guaranteed.

Cappé, Moulines, Rydén (2005) provide detailed convergence results for HMM-EM; Zucchini et al. emphasize practical convergence diagnostics.

---

## 5.3 Identifiability Theory

### 5.3.1 Definition of Identifiability

Let $\mathcal{P}_\theta$ be the joint distribution of $Y_{1:\infty}$ under parameter $\theta$. The HMM is **(strictly) identifiable** if
$$
\mathcal{P}_\theta = \mathcal{P}_{\theta'} \implies \theta' \in \mathcal{E}(\theta),
$$
where $\mathcal{E}(\theta)$ is the **equivalence class** of $\theta$ under state permutations (label switching).

Intuitively, **up to permutation of states**, the parameter is uniquely determined by the distribution of the observed process.

### 5.3.2 Simple Non-Identifiability Examples

- If two states have identical rows in $\boldsymbol{\Gamma}$ and identical emission parameters, merging them yields another parameter with the same observed distribution.
- If emission distributions are **linearly dependent** in certain ways (e.g. deterministic relationships), different combinations of transition probabilities and emissions can produce the same marginal process.

These examples show that identifiability requires **structural conditions**.

### 5.3.3 Sufficient Conditions for Finite-State HMMs (High-Level)

A line of work (e.g. Allman, Matias, Rhodes; Hsu, Kakade, Zhang; and results cited in Cappé et al.) gives sufficient conditions for identifiability of finite-state HMMs, typically requiring:

- The transition matrix $\boldsymbol{\Gamma}$ to be of **full rank** and ergodic;
- Emission distributions $f_i$ to be **distinct** and to span a sufficiently rich function space (e.g. a linearly independent set in $L^2$);
- Enough lags of the observed process to be considered.

Under such conditions, the joint distribution of $(Y_t, Y_{t+1}, Y_{t+2})$ (or higher blocks) contains enough information to recover $\theta$ up to permutation.

### 5.3.4 Practical Implications (Zucchini et al.)

In practice, Zucchini et al. stress that:

- One should avoid models where two states are effectively **indistinguishable** (same emissions, similar rows in $\boldsymbol{\Gamma}$);
- **Overly complex models** (too many states) can lead to weak identifiability and unstable estimates;
- State labels are arbitrary; interpretability often requires **post hoc ordering** or constraints.

---

## 5.4 Summary

In this section we:

- Defined MLE for HMMs and highlighted non-convexity and label switching;
- Derived the **Baum–Welch (EM) algorithm** from the complete-data likelihood, including explicit update formulas;
- Interpreted EM as **coordinate ascent** on an evidence lower bound, giving monotonicity and convergence to stationary points;
- Discussed **identifiability** and practical issues with overlapping states.

These results, together with the **asymptotic theory** in Section 6, provide a rigorous foundation for statistical inference in HMMs.

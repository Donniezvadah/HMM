# Section 7 – Non-Standard and Advanced Hidden Markov Models

This section surveys important **extensions and generalizations** of the basic finite-state HMM:

- **Continuous-state HMMs / state-space models** (Kalman filter as a linear-Gaussian HMM);
- **Nonparametric HMMs** with (theoretically) infinitely many states;
- **Switching state-space models** and regime-switching processes.

These models are beyond the core scope of Zucchini et al., but are natural continuations of the HMM framework.

---

## 7.1 Continuous-State HMMs and State-Space Models

### 7.1.1 General State-Space Models

A **state-space model** (SSM) generalizes finite-state HMMs by allowing the hidden state to live in a **continuous space**, typically \(\mathbb{R}^d\):

- Hidden process \((X_t)\) on \(\mathbb{R}^d\) with transition density
  \[
  p_\theta(x_{t+1} \mid x_t);
  \]
- Observation process \((Y_t)\) with conditional density
  \[
  g_\theta(y_t \mid x_t).
  \]

The Markov and conditional independence assumptions are analogous to HMMs:

- \(X_{t+1} \perp\!\!\perp X_{1:t-1} \mid X_t\);
- \(Y_t \perp\!\!\perp (X_{1:t-1}, X_{t+1:\infty}, Y_{1:t-1}, Y_{t+1:\infty}) \mid X_t\).

The joint density over \(X_{1:T}, Y_{1:T}\) factorizes as
\[
\mu(x_1) g(y_1 \mid x_1) \prod_{t=2}^T p(x_t \mid x_{t-1}) g(y_t \mid x_t),
\]
mirroring the finite-state HMM.

### 7.1.2 Linear-Gaussian State-Space Models (Kalman Filter)

A particularly important class is the **linear-Gaussian state-space model**:
\[
X_{t+1} = F X_t + W_t, \quad W_t \sim \mathcal{N}(0, Q),
\]
\[
Y_t = H X_t + V_t, \quad V_t \sim \mathcal{N}(0, R),
\]
where \(F, H\) are matrices, and \(Q, R\) are covariance matrices.

Here, \(X_t \in \mathbb{R}^d\) is a hidden **continuous state**, and \(Y_t \in \mathbb{R}^m\) is observed. The model is Gaussian and Markov; the **Kalman filter** provides exact filtering distributions
\[
\mathcal{L}(X_t \mid Y_{1:t}) = \mathcal{N}(m_t, P_t)
\]
via recursive updates of the mean \(m_t\) and covariance \(P_t\).

This is the continuous analog of the forward algorithm; see Douc, Moulines, Stoffer for a rigorous treatment.

### 7.1.3 Relation to Finite-State HMMs

Both finite-state HMMs and linear-Gaussian SSMs share:

- Markovian hidden dynamics;
- Conditional independence structure for observations;
- Recursive inference via **filtering/smoothing algorithms**.

Finite-state HMMs can be seen as a **discrete-state** special case of SSMs, while linear-Gaussian SSMs can be thought of as having a **continuous hidden state** with Gaussian transitions and emissions.

---

## 7.2 Nonparametric HMMs and Infinite-State Models

### 7.2.1 Motivation

Standard HMMs assume a **fixed number of states** \(K\). In some applications, choosing \(K\) is difficult or arbitrary. **Nonparametric HMMs** aim to allow a **potentially infinite** number of states, with the data effectively using only finitely many.

### 7.2.2 Dirichlet Process HMMs (Informal)

A **Dirichlet process (DP)** is a distribution over probability measures. In an HMM context, one can place a DP prior on the **rows** of the transition matrix, yielding a **DP-HMM**:

- Each row \(\boldsymbol{\Gamma}_{i,\cdot}\) is drawn from a DP centered on a base distribution over states;
- Posterior inference encourages **sparse** transition structures and can infer an effective number of states from data.

More structured models such as the **Hierarchical Dirichlet Process HMM (HDP-HMM)** share transition distributions across states and time.

The resulting posterior is supported on **countably infinite state spaces**, but in any finite dataset only a finite number of states have significant posterior mass.

### 7.2.3 Inference Challenges

Posterior inference in nonparametric HMMs typically requires:

- **Markov chain Monte Carlo (MCMC)** methods (Gibbs sampling, beam sampling);
- Or **variational inference** (truncating the infinite state space at a large \(K_{\max}\)).

While Zucchini et al. focus on finite-state models, the same **forward–backward structure** underlies these more complex Bayesian procedures.

---

## 7.3 Switching State-Space Models and Regime-Switching

### 7.3.1 Model Structure

A **switching state-space model** combines discrete regimes with continuous dynamics:

- Discrete hidden regime \(S_t \in \{1,\dots,K\}\) evolving as a Markov chain with transition matrix \(\boldsymbol{\Gamma}\);
- Continuous hidden state \(X_t \in \mathbb{R}^d\) with **regime-dependent dynamics**:
  \[
  X_{t+1} = F_{S_t} X_t + W_t, \quad W_t \sim \mathcal{N}(0, Q_{S_t});
  \]
- Observations
  \[
  Y_t = H_{S_t} X_t + V_t, \quad V_t \sim \mathcal{N}(0, R_{S_t}).
  \]

This yields a very flexible model where each regime has its own linear-Gaussian dynamics and observation structure.

### 7.3.2 Inference

Exact inference is generally **intractable** due to the exponential number of possible regime sequences and continuous states. Approaches include:

- **Approximate dynamic programming** (e.g. Gaussian sum approximations);
- **Particle filters** and **Rao–Blackwellized particle filters** that sample regime sequences while integrating over continuous states using Kalman filters;
- **EM-like algorithms** using approximate E-steps.

### 7.3.3 Applications

Switching and regime-switching models are common in:

- **Econometrics** (e.g. Markov-switching autoregressions for business cycles);
- **Signal processing** (systems with mode changes);
- **Engineering** (fault detection, hybrid systems).

They sit at the intersection of HMMs, state-space models, and control theory.

---

## 7.4 Summary

This section sketched several important generalizations of HMMs:

- **Continuous-state models** (state-space models) with Kalman filtering as a canonical example;
- **Nonparametric HMMs** with an unbounded number of states via Dirichlet process priors;
- **Switching state-space models** blending discrete regimes with continuous dynamics.

While Zucchini et al. primarily focus on finite-state HMMs, many of the **conceptual tools** carry over: Markov structure, conditional independence, and recursive inference algorithms.

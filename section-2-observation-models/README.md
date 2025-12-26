# Section 2 – Observation Models and Emission Processes

In an HMM, the hidden Markov chain \((S_t)_{t\ge 1}\) is not observed directly. Instead, we observe a process \((Y_t)_{t\ge 1}\), whose distribution is conditionally independent **given the hidden states**.

This section formalizes:

- The **graphical model** structure of HMMs;
- The **factorization** of the joint distribution;
- Classes of **emission distributions** (discrete, continuous, exponential family);
- Basic **identifiability** issues arising from emissions.

We follow the high-level view in Zucchini et al. (Chapter 2), but state the conditional independence structure more explicitly.

---

## 2.1 Conditional Independence Structure

### 2.1.1 Graphical Model Representation

An HMM with \(T\) time steps consists of:

- Hidden states \(S_1, \dots, S_T\) forming a Markov chain on \(\{1,\dots,K\}\);
- Observations \(Y_1, \dots, Y_T\) taking values in some space \(\mathcal{Y}\).

The **directed graphical model** has edges

- \(S_t \to S_{t+1}\) (hidden Markov chain);
- \(S_t \to Y_t\) (emission at each time).

The critical conditional independence assumptions are:

1. Given \(S_t\), the observation \(Y_t\) is **independent of all other states and observations**:
   \[
   Y_t \perp\!\!\perp \{S_s : s \ne t\}, \{Y_s : s \ne t\} \mid S_t.
   \]
2. The hidden chain is first-order Markov:
   \[
   S_{t+1} \perp\!\!\perp \{S_1,\dots,S_{t-1}\} \mid S_t.
   \]

Together, these imply a specific **factorization** of the joint distribution.

### 2.1.2 Factorization of the Joint Distribution

Let \(s_{1:T} = (s_1,\dots,s_T)\) and \(y_{1:T} = (y_1,\dots,y_T)\). The joint distribution of states and observations factorizes as
\[
\mathbb{P}(S_{1:T} = s_{1:T}, Y_{1:T} = y_{1:T})
= \delta_{s_1} \, f_{s_1}(y_1) \prod_{t=2}^T \gamma_{s_{t-1}, s_t} \, f_{s_t}(y_t),
\]
where

- \(\boldsymbol{\delta} = (\delta_i)\) is the initial distribution \(\mathbb{P}(S_1 = i)\);
- \(\boldsymbol{\Gamma} = (\gamma_{ij})\) is the transition matrix \(\mathbb{P}(S_t = j \mid S_{t-1} = i)\);
- \(f_i(\cdot)\) is the **emission density or mass function** for state \(i\).

This is the basic factorization that Zucchini et al. use throughout their book; it underlies all efficient algorithms (forward–backward, Viterbi, EM).

### 2.1.3 d-Separation and Conditional Independences

The graphical structure immediately yields many conditional independences via **d-separation**:

- Given \(S_t\), the past and future observations are conditionally independent:
  \[
  Y_{1:t-1} \perp\!\!\perp Y_{t+1:T} \mid S_t.
  \]
- Given the full state sequence \(S_{1:T}\), the observations are conditionally independent across time:
  \[
  Y_t \perp\!\!\perp Y_s \mid S_{1:T}, \quad t \ne s.
  \]
- Given **all observations** \(Y_{1:T}\), the hidden states form a **Markov random field** (an undirected chain), but conditional dependences become more complex.

Understanding these independences helps in designing **approximate inference algorithms** and **variational factorizations**.

---

## 2.2 Emission Distributions

### 2.2.1 Discrete Emissions

If \(Y_t\) takes values in a finite or countable set \(\mathcal{Y} = \{1,\dots,M\}\), each state \(i\) has a probability mass function
\[
\mathbb{P}(Y_t = y \mid S_t = i) = b_i(y), \quad y \in \mathcal{Y},
\]
with \(b_i(y) \ge 0\) and \(\sum_y b_i(y) = 1\).

Collect \(b_i\) into an **emission matrix** \(\mathbf{B}\) of size \(K \times M\), where \(B_{iy} = b_i(y)\). Discrete-emission HMMs are the classical setting in **speech recognition** and many applications in **bioinformatics**.

In Zucchini et al., discrete emissions appear in introductory examples and in categorical time series modeling.

### 2.2.2 Continuous Emissions

If \(Y_t\) takes values in \(\mathbb{R}^d\) (or a subset), each state \(i\) has a **density** (with respect to Lebesgue measure) \(f_i(y)\), so that
\[
\mathbb{P}(Y_t \in A \mid S_t = i)
= \int_A f_i(y) \, dy.
\]

Common parametric choices:

- **Gaussian emissions:** \(f_i(y) = \mathcal{N}(y; \mu_i, \Sigma_i)\);
- **Mixtures of Gaussians:** to increase flexibility;
- Other **exponential family** densities (see next subsection).

Continuous-emission HMMs are heavily treated in Zucchini et al. for modeling **time series of real-valued measurements** (e.g. environmental data, financial returns).

### 2.2.3 Exponential Family Emissions

Many emission models fall into the **exponential family**. A density (or mass function) \(f(y;\eta)\) is in an exponential family if it can be written as
\[
f(y; \eta) = h(y) \exp\{ \langle \eta, T(y) \rangle - A(\eta) \},
\]
where

- \(T(y)\) is the vector of **sufficient statistics**;
- \(\eta\) is the **natural parameter**;
- \(A(\eta)\) is the **log-partition function** ensuring normalization;
- \(h(y)\) is the base measure or carrier density.

In an HMM with exponential-family emissions, each state \(i\) has its own natural parameter \(\eta_i\), and thus its own emission distribution \(f_i(y)\). This structure simplifies:

- Derivation of **EM (Baum–Welch) updates** for emission parameters;
- Computation of gradients and Fisher information.

The connection to **information geometry** (Section 0.3) arises because the log-partition function \(A(\eta)\) is the **convex conjugate** of negative entropy, and KL divergence between two exponential-family members has a natural Bregman form.

### 2.2.4 Identifiability Issues

**Identifiability** asks whether the parameter \(\theta\) of an HMM (transition matrix, emissions, etc.) is uniquely determined by the distribution of \(Y_{1:T}\) (for all \(T\) large enough), up to label permutations of the hidden states.

Even with rich emission families, several issues arise:

- **Label switching:** If we permute state indices, say swap states 1 and 2, and correspondingly permute rows/columns of \(\boldsymbol{\Gamma}\) and emission parameters, the distribution of \(Y_{1:T}\) is unchanged. Thus, identifiability is at best **up to permutation**.
- **Overlapping emissions:** If two states share identical emission distributions (e.g. \(f_1 = f_2\)) and transition rows, they may be **indistinguishable**.
- **Non-identifiability in mixtures:** In some cases, different combinations of transition probabilities and emission parameters can yield the same observed process distribution.

The formal theory of identifiability in HMMs is nontrivial (see Section 5.3 and references there). Zucchini et al. discuss practical implications: e.g., in estimation, one must be aware that state labels are arbitrary and that some parameter settings may be weakly identified.

---

## 2.3 Observation Models in Practice (Zucchini et al.)

Zucchini et al. provide many concrete observation models:

- **Count data:** Poisson or negative binomial emissions for counts (e.g. number of events per time unit);
- **Continuous data:** Gaussian or t-distributed emissions for real-valued series;
- **Circular data:** von Mises or wrapped distributions for angles;
- **Multivariate data:** multivariate normal or copula-based constructions.

In each case, the key is to specify, for each state \(i\), a parametric family
\[
\{ f_i(\cdot; \phi_i) : \phi_i \in \Phi_i \}
\]
and then estimate \(\phi_i\) jointly with \(\boldsymbol{\delta}\) and \(\boldsymbol{\Gamma}\) (typically by maximum likelihood using EM).

---

## 2.4 Summary and Outlook

By now you should understand:

- How the **conditional independence structure** of HMMs induces a specific **factorization** of the joint distribution;
- The role of **emission distributions** in shaping the model’s expressiveness;
- Basic **identifiability concerns** arising from overlapping or non-distinct emissions.

These ideas feed directly into:

- **Section 3:** Formal definition of HMMs and likelihood factorization;
- **Section 4:** Algorithms for computing marginal and conditional distributions over states given observations;
- **Section 5:** Parameter estimation (MLE, EM) and identifiability theory.

For additional reading:

- Zucchini et al., Chapters 2–3 (construction of HMMs and emission models);
- Cappé, Moulines, Rydén (2005), Chapters 1–2 (measure-theoretic HMM definition and basic properties).

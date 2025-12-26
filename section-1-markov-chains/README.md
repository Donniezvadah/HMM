# Section 1 – Markov Chains (Fully Rigorous)

This section develops the **Markov chain theory** that underlies finite-state HMMs. We focus on:

- Finite-state **homogeneous Markov chains** and their invariant distributions;
- **Ergodic properties**: irreducibility, aperiodicity, mixing, spectral gap;
- A brief look at **non-homogeneous** chains, which appear in some generalized HMMs.

Zucchini et al. treat finite-state Markov chains at an applied level; here we give a more rigorous account compatible with their notation.

---

## 1.1 Finite-State Markov Chains

### 1.1.1 Definition and Transition Kernels

Let the **state space** be $E = \{1,\dots,K\}$. A stochastic process $(S_t)_{t \ge 1}$ with values in $E$ is a (time-homogeneous) **Markov chain** with transition matrix $\boldsymbol{\Gamma} = (\gamma_{ij})$ if
$$
\mathbb{P}(S_{t+1} = j \mid S_1, \dots, S_t) = \mathbb{P}(S_{t+1} = j \mid S_t) = \gamma_{S_t j}, \quad \forall t \ge 1.
$$

Equivalently, for any sequence $i_1, \dots, i_T$ in $E$, the joint probability is
$$
\mathbb{P}(S_1 = i_1, \dots, S_T = i_T)
= \delta_{i_1} \prod_{t=1}^{T-1} \gamma_{i_t i_{t+1}},
$$
where $\boldsymbol{\delta} = (\delta_i)$ is the **initial distribution** $\delta_i = \mathbb{P}(S_1 = i)$.

This is exactly the hidden-state dynamics that Zucchini et al. use to define finite-state HMMs; the HMM adds an **observation process** on top of this chain.

### 1.1.2 Chapman–Kolmogorov Equations

Let $\boldsymbol{\Gamma}^{(n)}$ denote the $n$-step transition matrix, with entries
$$
\gamma^{(n)}_{ij} = \mathbb{P}(S_{t+n} = j \mid S_t = i).
$$

Then the **Chapman–Kolmogorov equations** state that for all $m,n \ge 0$,
$$
\boldsymbol{\Gamma}^{(m+n)} = \boldsymbol{\Gamma}^{(m)} \boldsymbol{\Gamma}^{(n)}.
$$

In particular, $\boldsymbol{\Gamma}^{(n)} = \boldsymbol{\Gamma}^n$ (the usual matrix power). This ties Markov chain evolution directly to the spectral properties of $\boldsymbol{\Gamma}$.

### 1.1.3 Stationary and Invariant Distributions

A probability vector $\boldsymbol{\pi} \in \Delta^{K-1}$ is a **stationary distribution** for $\boldsymbol{\Gamma}$ if
$$
\boldsymbol{\pi}^\top \boldsymbol{\Gamma} = \boldsymbol{\pi}^\top.
$$

Interpretation:

- If $S_1 \sim \boldsymbol{\pi}$, then $S_t \sim \boldsymbol{\pi}$ for all $t$; the chain is **in equilibrium**.
- If the chain is **irreducible and aperiodic**, $\boldsymbol{\pi}$ is **unique**, and the distribution of $S_t$ converges to $\boldsymbol{\pi}$ for any initial distribution $\boldsymbol{\delta}$.

The existence and uniqueness of $\boldsymbol{\pi}$ are guaranteed by **Perron–Frobenius theory** (Section 0.2) for irreducible, aperiodic stochastic matrices.

### 1.1.4 Reversibility and Detailed Balance

A Markov chain with transition matrix $\boldsymbol{\Gamma}$ and stationary distribution $\boldsymbol{\pi}$ is **reversible** if it satisfies the **detailed balance equations**
$$
\pi_i \, \gamma_{ij} = \pi_j \, \gamma_{ji}, \quad \forall i,j.
$$

Intuitively, under stationarity, the probability flow from $i$ to $j$ equals that from $j$ to $i$.

Consequences:

- In the inner product space $L^2(\boldsymbol{\pi})$, $\boldsymbol{\Gamma}$ is **self-adjoint**:
  $$
  \langle f, \boldsymbol{\Gamma} g \rangle_\pi = \langle \boldsymbol{\Gamma} f, g \rangle_\pi
  $$
  for functions $f,g : E \to \mathbb{R}$, where $\langle f,g \rangle_\pi = \sum_i \pi_i f(i) g(i)$.
- Hence, the spectrum of $\boldsymbol{\Gamma}$ is **real**, and spectral analysis is particularly transparent.

In HMMs, even if the hidden chain is not assumed reversible, reversible chains are a useful class for **examples**, **counterexamples**, and **mixing-time calculations**.

---

## 1.2 Ergodic Theory of Markov Chains

### 1.2.1 Irreducibility and Communication Classes

For states $i,j \in E$, write $i \rightsquigarrow j$ if there exists $n \ge 0$ such that $\gamma^{(n)}_{ij} > 0$ (a path of positive probability from $i$ to $j$). We say **$i$ communicates with $j$**, written $i \leftrightarrow j$, if both $i \rightsquigarrow j$ and $j \rightsquigarrow i$ hold.

This is an equivalence relation, partitioning $E$ into **communicating classes**. A chain is **irreducible** if it has a single communicating class (every state communicates with every other).

In HMMs, irreducibility of the hidden chain ensures that every state can eventually be reached from any other, which is important for:

- Existence and uniqueness of a stationary distribution;
- Identifiability and mixing assumptions in asymptotic theory (Section 6).

### 1.2.2 Periodicity and Aperiodicity

The **period** of a state $i$ is
$$
\mathrm{per}(i) = \gcd\{ n \ge 1 : \gamma^{(n)}_{ii} > 0 \}.
$$

In an irreducible chain, all states share the same period, so we can speak of **the** period of the chain. A chain is **aperiodic** if $\mathrm{per}(i) = 1$ for some (hence all) $i$.

Aperiodicity rules out deterministic cycles and is necessary for convergence of $\mathbb{P}(S_t = \cdot)$ to $\boldsymbol{\pi}$ in total variation.

### 1.2.3 Ergodic Theorem for Finite-State Markov Chains

Let $(S_t)$ be irreducible and aperiodic with stationary distribution $\boldsymbol{\pi}$. Then for any bounded function $f : E \to \mathbb{R}$,
$$
\frac{1}{T} \sum_{t=1}^T f(S_t) \xrightarrow[T\to\infty]{\text{a.s.}} \sum_{i=1}^K \pi_i f(i) =: \mathbb{E}_\pi[f(S)].
$$

This is the **ergodic theorem**: time averages converge almost surely to space averages under $\boldsymbol{\pi}$. It is a Markov-chain version of the **strong law of large numbers**.

In HMMs, ergodic theorems are used to prove **consistency of estimators** and to analyze limiting behavior of likelihoods per unit time.

### 1.2.4 Mixing Times and Total Variation Distance

For a probability vector $\boldsymbol{\mu}$ on $E$, the **total variation distance** to $\boldsymbol{\pi}$ is
$$
\lVert \boldsymbol{\mu} - \boldsymbol{\pi} \rVert_{\mathrm{TV}}
= \frac{1}{2} \sum_{i=1}^K |\mu_i - \pi_i|.
$$

Let $\boldsymbol{\mu}_t = \boldsymbol{\delta}^\top \boldsymbol{\Gamma}^t$ be the distribution of $S_t$ starting from $\boldsymbol{\delta}$. The **mixing time** $t_{\mathrm{mix}}(\varepsilon)$ is
$$
 t_{\mathrm{mix}}(\varepsilon) = \min\Bigl\{ t : \sup_{\boldsymbol{\delta}} \lVert \boldsymbol{\mu}_t - \boldsymbol{\pi} \rVert_{\mathrm{TV}} \le \varepsilon \Bigr\}.
$$

In finite-state irreducible aperiodic chains, $t_{\mathrm{mix}}(\varepsilon) < \infty$ for all $\varepsilon > 0$. Spectral methods and coupling (next subsection) give quantitative bounds.

### 1.2.5 Spectral Gap and Convergence Rates

Suppose the chain is reversible with respect to $\boldsymbol{\pi}$, with eigenvalues of $\boldsymbol{\Gamma}$ ordered as
$$
1 = \lambda_1 > \lambda_2 \ge \dots \ge \lambda_K > -1.
$$

The **spectral gap** is $\gamma = 1 - \lambda_2$. One can show (see e.g. books on Markov chain mixing) that
$$
\lVert \boldsymbol{\mu}_t - \boldsymbol{\pi} \rVert_{\mathrm{TV}}
\le C \, (1-\gamma)^t
$$
for some constant $C$ depending on $\boldsymbol{\delta}$. Thus, a larger spectral gap implies faster convergence to stationarity.

In HMMs, these spectral-gap-based bounds transfer to **stability of the filtering distribution**: the distribution of $S_t$ given observations becomes asymptotically independent of the initial distribution.

### 1.2.6 Coupling Arguments (Sketch)

A powerful probabilistic technique for bounding mixing times is **coupling**: construct two copies of the chain, $(S_t)$ and $(S'_t)$, possibly dependent, such that

- Marginally, each evolves according to $\boldsymbol{\Gamma}$;
- They eventually **coalesce**: $S_t = S'_t$ for all sufficiently large $t$.

Define the **coupling time**
$$
T_c = \inf\{ t \ge 0 : S_t = S'_t \}.
$$

Then for any initial distributions $\boldsymbol{\delta}, \boldsymbol{\delta}'$,
$$
\lVert \boldsymbol{\mu}_t - \boldsymbol{\mu}'_t \rVert_{\mathrm{TV}}
\le \mathbb{P}(T_c > t).
$$

Hence, controlling $\mathbb{P}(T_c > t)$ yields mixing bounds. The idea of coupling will reappear implicitly in **filter stability** results in HMMs.

---

## 1.3 Non-Homogeneous Markov Chains

In some extensions of HMMs, the hidden state process may have **time-varying transitions**, represented by a sequence of stochastic matrices $(\boldsymbol{\Gamma}_t)$. Then
$$
\mathbb{P}(S_{t+1} = j \mid S_t = i) = (\boldsymbol{\Gamma}_t)_{ij}.
$$

### 1.3.1 Product of Time-Varying Kernels

Define the $n$-step transition kernel from time $t$ to $t+n$ as
$$
\boldsymbol{\Gamma}_{t, t+n} = \boldsymbol{\Gamma}_t \boldsymbol{\Gamma}_{t+1} \cdots \boldsymbol{\Gamma}_{t+n-1}.
$$

The analog of Chapman–Kolmogorov holds in the obvious way:
$$
\boldsymbol{\Gamma}_{t, t+m+n} = \boldsymbol{\Gamma}_{t, t+m} \boldsymbol{\Gamma}_{t+m, t+m+n}.
$$

### 1.3.2 Stability Conditions

Without time-homogeneity, there may be **no stationary distribution**. Instead, one studies **stability** and **ergodicity** via conditions such as:

- Uniform **Doeblin conditions** (lower bounds on transition probabilities);
- **Dobrushin contraction coefficients** ensuring that products of kernels contract distances between probability distributions.

These ideas become particularly relevant when considering **non-stationary HMMs** or **online learning** settings (see Section 9).

---

## 1.4 Connection to HMMs and Zucchini et al.

In Zucchini et al., the hidden process $(S_t)$ of an HMM is always a **finite-state Markov chain** with transition matrix $\boldsymbol{\Gamma}$ and initial distribution $\boldsymbol{\delta}$. The properties introduced here feed directly into later sections:

- **Section 3:** Uses the Markov property to factorize the joint HMM likelihood;
- **Section 4:** Forward–backward and Viterbi algorithms exploit $\boldsymbol{\Gamma}$ as the transition kernel;
- **Section 6:** Ergodicity and mixing of $(S_t)$ underpin **consistency** and **CLTs** for estimators.

For more detailed Markov chain theory in a measure-theoretic style, see:

- Cappé, Moulines, Rydén (2005), Chapters 1–2;
- Douc, Moulines, Stoffer (2014), Chapters 2–3.

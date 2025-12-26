# Section 0 – Mathematical Prerequisites for Hidden Markov Models

This section collects the **mathematical foundations** required for a rigorous treatment of Hidden Markov Models (HMMs). The goal is **not** to teach full measure-theoretic probability from scratch, but to make precise the pieces that will be used repeatedly later.

Throughout, we aim to be compatible with the notation and level of **Zucchini, MacDonald, Langrock** ("Zucchini et al.") while pushing the theory somewhat further when needed for Sections 4–6.

---

## 0.1 Measure-Theoretic Probability (Light but Precise)

### 0.1.1 Probability Spaces

A **probability space** is a triple \((\Omega, \mathcal{F}, \mathbb{P})\) where

- \(\Omega\) is the **sample space** (set of outcomes);
- \(\mathcal{F} \subseteq 2^{\Omega}\) is a **\(\sigma\)-algebra** of events (closed under complements and countable unions);
- \(\mathbb{P} : \mathcal{F} \to [0,1]\) is a **probability measure** with \(\mathbb{P}(\Omega)=1\) and countable additivity.

For HMMs we usually work with products of measurable spaces, e.g. sequences of states and observations. The relevant product \(\sigma\)-algebras and measures are:

- For a measurable space \((S, \mathcal{S})\), the **countable product** \((S^{\mathbb{N}}, \mathcal{S}^{\otimes \mathbb{N}})\) is defined via the smallest \(\sigma\)-algebra making all coordinate projections measurable.
- For a Markov chain \((S_t)_{t \ge 1}\), the joint law of the whole sequence lives on such a product space.

In finite-state HMMs, \(S = \{1,\dots,K\}\) with the **discrete \(\sigma\)-algebra** (all subsets), so measurability is trivial; nevertheless, the measure-theoretic formulation clarifies **conditional expectations** and **ergodic theorems** later.

### 0.1.2 Random Variables and Distributions

A **random variable** with values in a measurable space \((S, \mathcal{S})\) is a measurable map
\[
X : (\Omega, \mathcal{F}) \to (S, \mathcal{S}).
\]

The **distribution** (or law) of \(X\) is the pushforward measure \(\mathbb{P}_X\) on \((S, \mathcal{S})\):
\[
\mathbb{P}_X(A) = \mathbb{P}(X \in A), \quad A \in \mathcal{S}.
\]

In HMMs we will consider random variables \(S_t\) (hidden states) and \(Y_t\) (observations). Their joint distribution factorizes in a special way due to the **Markov property** and **conditional independence**, which we will formalize later.

### 0.1.3 Expectation and Conditional Expectation

For an integrable real-valued random variable \(X\), its **expectation** is
\[
\mathbb{E}[X] = \int_{\Omega} X(\omega) \, \mathbb{P}(d\omega),
\]

or equivalently, if \(X\) takes values in \(\mathbb{R}\) with distribution \(\mu = \mathbb{P}_X\),
\[
\mathbb{E}[X] = \int_{\mathbb{R}} x \, \mu(dx).
\]

For a sub-\(\sigma\)-algebra \(\mathcal{G} \subseteq \mathcal{F}\), the **conditional expectation** of \(X\) given \(\mathcal{G}\) is a \(\mathcal{G}\)-measurable random variable \(\mathbb{E}[X\mid \mathcal{G}]\) such that
\[
\int_G \mathbb{E}[X\mid\mathcal{G}] \, d\mathbb{P} = \int_G X \, d\mathbb{P}, \quad \forall G \in \mathcal{G}.
\]

Key properties (used constantly in HMM derivations):

- **Linearity:** \(\mathbb{E}[aX + bY \mid \mathcal{G}] = a\,\mathbb{E}[X\mid\mathcal{G}] + b\,\mathbb{E}[Y\mid\mathcal{G}]\).
- **Tower property:** If \(\mathcal{H} \subseteq \mathcal{G} \subseteq \mathcal{F}\), then
  \[
  \mathbb{E}[\mathbb{E}[X\mid\mathcal{G}]\mid \mathcal{H}] = \mathbb{E}[X\mid\mathcal{H}].
  \]
- **Taking out what is known:** If \(Z\) is \(\mathcal{G}\)-measurable and integrable,
  \[
  \mathbb{E}[ZX\mid\mathcal{G}] = Z\,\mathbb{E}[X\mid\mathcal{G}].
  \]

In HMMs, filtering and smoothing can be viewed as **computing conditional expectations** like \(\mathbb{E}[g(S_t) \mid Y_{1:T}]\) for suitable functions \(g\). The forward–backward algorithms are efficient implementations of these operations.

### 0.1.4 Regular Conditional Probabilities

Given random variables \(X\) and \(Y\) on a probability space, a **regular conditional probability** of \(X\) given \(Y=y\) is a family of probability measures \(\{\mathbb{P}(X \in \cdot \mid Y=y)\}\) such that

- For each measurable \(A\), the map \(y \mapsto \mathbb{P}(X \in A \mid Y=y)\) is measurable;
- For each measurable \(B\),
  \[
  \mathbb{P}(X \in B, Y \in C) = \int_C \mathbb{P}(X \in B \mid Y=y) \, \mathbb{P}_Y(dy).
  \]

On **standard Borel spaces** (Polish spaces with their Borel \(\sigma\)-algebra), regular conditional probabilities always exist and are unique up to \(\mathbb{P}_Y\)-null sets. This justifies writing objects like
\[
\mathbb{P}(S_t = i \mid Y_{1:T}=y_{1:T})
\]
rigorously, which is what the forward–backward algorithms compute.

### 0.1.5 Modes of Convergence

We briefly recall three notions of convergence for a sequence of random variables \((X_n)\):

- **Almost sure (a.s.) convergence:** \(X_n \to X\) a.s. if
  \[
  \mathbb{P}\bigl(\{\omega : X_n(\omega) \to X(\omega)\}\bigr) = 1.
  \]
- **Convergence in probability:** \(X_n \to X\) in probability if, for all \(\varepsilon > 0\),
  \[
  \lim_{n\to\infty} \mathbb{P}(|X_n - X| > \varepsilon) = 0.
  \]
- **\(L^p\) convergence:** \(X_n \to X\) in \(L^p\) (for \(p \ge 1\)) if
  \[
  \lim_{n\to\infty} \mathbb{E}[|X_n - X|^p] = 0.
  \]

For asymptotic theory in HMMs (Section 6), we will need **laws of large numbers** and **central limit theorems** for functionals of an ergodic Markov chain. These are typically stated in terms of convergence in probability or distribution, and proved using almost sure convergence plus dominated convergence.

---

## 0.2 Linear Algebra and Spectral Theory

### 0.2.1 Probability Vectors and the Simplex

For a finite state space of size \(K\), a **probability vector** is
\[
\boldsymbol{\mu} = (\mu_1, \dots, \mu_K)^\top, \quad \mu_i \ge 0, \quad \sum_{i=1}^K \mu_i = 1.
\]

The set of all such vectors is the **probability simplex**
\[
\Delta^{K-1} = \Bigl\{ \boldsymbol{\mu} \in \mathbb{R}^K : \mu_i \ge 0, \sum_i \mu_i = 1 \Bigr\}.
\]

We measure distances on \(\Delta^{K-1}\) using norms:

- **\(\ell^1\) norm:** \(\lVert \mu - \nu \rVert_1 = \sum_i |\mu_i - \nu_i|\) (twice the total variation distance);
- **\(\ell^2\) norm:** \(\lVert \mu - \nu \rVert_2 = (\sum_i (\mu_i - \nu_i)^2)^{1/2}\).

Both will appear in mixing-time and stability results for Markov chains and filters.

### 0.2.2 Stochastic Matrices

A **row-stochastic matrix** is a \(K \times K\) matrix \(\boldsymbol{\Gamma} = (\gamma_{ij})\) with
\[
\gamma_{ij} \ge 0, \quad \sum_{j=1}^K \gamma_{ij} = 1 \quad \text{for all } i.
\]

In finite-state HMMs (following Zucchini et al.), \(\boldsymbol{\Gamma}\) denotes the **transition matrix** of the hidden Markov chain \((S_t)\):
\[
\gamma_{ij} = \mathbb{P}(S_{t+1} = j \mid S_t = i).
\]

Given a probability vector \(\boldsymbol{\mu}\), the product \(\boldsymbol{\mu}^\top \boldsymbol{\Gamma}\) is again a probability vector, representing the distribution of \(S_{t+1}\) if \(\boldsymbol{\mu}\) is the distribution of \(S_t\).

### 0.2.3 Perron–Frobenius Theory

For a **non-negative matrix** \(A \in \mathbb{R}^{K \times K}\) (i.e. \(A_{ij} \ge 0\)), the Perron–Frobenius theorem gives powerful spectral properties. In particular, if \(A\) is **irreducible**, then

- There exists a **positive eigenvalue** \(\rho(A) > 0\) (the spectral radius) with a corresponding **positive eigenvector** \(v > 0\).
- \(\rho(A)\) is **simple** (algebraic multiplicity 1), and no other eigenvector with non-negative entries exists for a different eigenvalue.

For a **stochastic matrix** \(\boldsymbol{\Gamma}\):

- Its spectral radius satisfies \(\rho(\boldsymbol{\Gamma}) = 1\), since \(\boldsymbol{\Gamma}\mathbf{1} = \mathbf{1}\).
- If \(\boldsymbol{\Gamma}\) is irreducible and aperiodic, the **left eigenvector** corresponding to eigenvalue 1, normalized to sum to 1, is the **unique stationary distribution** \(\boldsymbol{\pi}\):
  \[
  \boldsymbol{\pi}^\top \boldsymbol{\Gamma} = \boldsymbol{\pi}^\top.
  \]

This provides the spectral foundation for **ergodicity** of finite-state Markov chains, and later for stability of HMM filters.

### 0.2.4 Spectral Gap and Convergence Rates

Let the eigenvalues of a stochastic matrix \(\boldsymbol{\Gamma}\) be ordered as
\[
1 = \lambda_1 > |\lambda_2| \ge \dots \ge |\lambda_K|.
\]

The **spectral gap** is
\[
\gamma := 1 - |\lambda_2|.
\]

For many chains (especially reversible ones), the convergence of \(\boldsymbol{\mu}_0^\top \boldsymbol{\Gamma}^t\) to the stationary distribution \(\boldsymbol{\pi}^\top\) in \(\ell^2\) or total variation can be bounded in terms of \(\gamma\). Roughly,
\[
\lVert \boldsymbol{\mu}_0^\top \boldsymbol{\Gamma}^t - \boldsymbol{\pi}^\top \rVert_2 \le C (1-\gamma)^t.
\]

More precise inequalities follow from the spectral decomposition of \(\boldsymbol{\Gamma}\) and, in the reversible case, from its self-adjointness in \(L^2(\boldsymbol{\pi})\).

These ideas will underpin **mixing-time** and **filter stability** results (Sections 1.2 and 4.1).

---

## 0.3 Optimization and Information Geometry

### 0.3.1 Convexity on the Probability Simplex

A function \(f : \Delta^{K-1} \to \mathbb{R}\) is **convex** if
\[
f(\theta \mu + (1-\theta)\nu) \le \theta f(\mu) + (1-\theta) f(\nu)
\]
for all \(\mu, \nu \in \Delta^{K-1}\) and \(\theta \in [0,1]\).

Many information-theoretic functionals are convex or strictly convex on \(\Delta^{K-1}\). Examples:

- Negative entropy \(H(\mu) = -\sum_i \mu_i \log \mu_i\) is **strictly concave**;
- The **Kullback–Leibler divergence** (KL) is jointly convex in \((p,q)\).

Convexity is central in understanding **EM updates**, **variational approximations**, and the geometry of the **log-likelihood surface** in HMMs.

### 0.3.2 Kullback–Leibler Divergence as a Bregman Divergence

For two discrete distributions \(p,q \in \Delta^{K-1}\) with full support (\(p_i, q_i > 0\)), the **KL divergence** is
\[
\mathrm{KL}(p \Vert q) = \sum_{i=1}^K p_i \log\frac{p_i}{q_i}.
\]

KL divergence can be written as a **Bregman divergence** associated with the **negative entropy** function
\[
\phi(p) = \sum_i p_i \log p_i.
\]

The Bregman divergence generated by \(\phi\) is
\[
D_\phi(p,q) = \phi(p) - \phi(q) - \langle \nabla \phi(q), p - q \rangle,
\]
where \(\langle \cdot,\cdot \rangle\) is the usual inner product on \(\mathbb{R}^K\). A straightforward calculation shows
\[
D_\phi(p,q) = \mathrm{KL}(p \Vert q).
\]

This interpretation highlights several facts:

- \(\mathrm{KL}(p \Vert q) \ge 0\) with equality iff \(p=q\) (strict convexity of \(\phi\));
- KL is **asymmetric**, unlike a metric, which shapes the geometry of likelihood-based optimization.

In HMMs, KL divergence arises when analyzing **consistency** and **information projections**, and in understanding why the EM algorithm can be seen as **coordinate ascent on a lower bound** involving KL terms.

### 0.3.3 Duality and Entropy-Regularized Problems

Given a convex function \(\phi\), its **convex conjugate** \(\phi^*\) is
\[
\phi^*(y) = \sup_{x} \{ \langle x, y \rangle - \phi(x) \}.
\]

For \(\phi(p) = \sum_i p_i \log p_i\) (negative entropy), \(\phi^*\) is the log-partition function
\[
\phi^*(\eta) = \log \sum_i e^{\eta_i}.
\]

This duality underlies the **exponential family** structure of many emission distributions (Section 2.2) and appears in **variational formulations** of inference in HMMs:

- Entropy-regularized objectives of the form
  \[
  \max_{q} \Big\{ \mathbb{E}_q[\log p(Y,S)] + H(q) \Big\}
  \]
  lead to exponential-family solutions for the optimal \(q\).

In the context of Zucchini et al., this background explains why **log-sum-exp** expressions appear in marginal likelihoods and why certain optimization problems have tractable, closed-form updates.

---

## 0.4 Summary and Connection to Later Sections

After this section, you should be comfortable with:

- **Probability spaces, random variables, and conditional expectations** in a measure-theoretic language;
- **Finite-dimensional linear algebra** for stochastic matrices, including Perron–Frobenius theory and spectral gaps;
- **Basic convex analysis** on probability simplices, and the interpretation of **KL divergence as a Bregman divergence**.

These tools will be used heavily in:

- **Section 1:** rigorous Markov chain theory (ergodicity, mixing);
- **Section 3–4:** derivation and correctness proofs of forward–backward and Viterbi algorithms;
- **Section 5–6:** EM algorithm analysis, identifiability, consistency, and asymptotic normality.

For a softer introduction, you may cross-reference:

- Zucchini et al., Chapters 1–2, for probabilistic notation and basic Markov chain ideas;
- Murphy (2012), Chapters 2–3, for probability and exponential families;
- Cappé, Moulines, Rydén (2005), Chapter 1, for a more advanced measure-theoretic setup.

# Section 6 – Asymptotics and Statistical Theory for HMMs

This section treats the **large-sample behavior** of estimators in Hidden Markov Models, focusing on:

- **Consistency** of the maximum likelihood estimator (MLE);
- **Asymptotic normality** and Fisher information;
- **Misspecification** and pseudo-true parameters.

The development is inspired by **Cappé, Moulines, Rydén (2005)** and **Douc, Moulines, Stoffer (2014)**, who provide a rigorous ergodic-theoretic foundation. Zucchini et al. present the main ideas informally; here we state them more precisely.

We mainly consider **finite-state HMMs** with emission densities \(f_i\) that are smooth in parameters.

---

## 6.1 Setup and Regularity Conditions

Let \(\{(S_t,Y_t)\}_{t\ge1}\) be an HMM with true parameter \(\theta^*\). Assume:

1. The hidden chain \((S_t)\) is **irreducible and aperiodic**, with unique stationary distribution \(\boldsymbol{\pi}^*\);
2. Under \(\theta^*\), the joint process \((S_t,Y_t)\) is **stationary and ergodic** (true if we start from stationarity or after a transient);
3. The parameter space \(\Theta\) is compact or the log-likelihood is **coercive**;
4. The emission densities \(f_i(y;\phi_i)\) and transition probabilities are **smooth** in \(\theta\);
5. The model is **identifiable up to permutation** (Section 5.3).

We observe \(Y_{1:T}\) and compute the MLE \(\hat{\theta}_T\).

---

## 6.2 Consistency of the MLE

### 6.2.1 Log-Likelihood per Observation

Define the **average log-likelihood**
\[
\bar{\ell}_T(\theta) = \frac{1}{T} \ell_T(\theta) = \frac{1}{T} \log p_\theta(Y_{1:T}).
\]

A key result: for each fixed \(\theta\), the limit
\[
\ell_\infty(\theta) = \lim_{T\to\infty} \bar{\ell}_T(\theta)
\]
exists **almost surely** (and in \(L^1\)), and can be expressed as an expectation under the stationary distribution of the hidden chain and emissions.

This follows from subadditive ergodic theorems or from explicit Markov chain arguments (see Cappé et al., Chapter 9).

### 6.2.2 Identification of the Limit

Under stationarity, one can show that
\[
\ell_\infty(\theta)
= \mathbb{E}_{\theta^*}\big[ \log p_\theta(Y_0 \mid Y_{-\infty:-1}) \big],
\]
where \(Y_{-\infty:0}\) denotes the infinite past.

Intuitively, \(\ell_\infty(\theta)\) is the **expected log predictive likelihood** of \(Y_0\) given the entire past, under the true parameter \(\theta^*\), but evaluated at a candidate parameter \(\theta\).

### 6.2.3 Consistency under Correct Specification

If the model is correctly specified and identifiable (up to permutation), then
\[
\ell_\infty(\theta) \le \ell_\infty(\theta^*)
\]
with equality **only** if \(\theta\) belongs to the permutation-equivalence class of \(\theta^*\).

Under mild regularity conditions, we can show that
\[
\sup_{\theta \in \Theta} \bar{\ell}_T(\theta)
\xrightarrow[T\to\infty]{\text{a.s.}} \sup_{\theta \in \Theta} \ell_\infty(\theta) = \ell_\infty(\theta^*).
\]

If the argmax of \(\ell_\infty\) is unique up to permutation, then **any sequence of MLEs** \(\hat{\theta}_T\) converges almost surely to the equivalence class of \(\theta^*\). This is **strong consistency** (modulo label switching).

### 6.2.4 Misspecification and Pseudo-True Parameters

If the true data-generating process is **not** in the model class, there is no \(\theta^*\) such that \(\mathcal{P}_\theta = \mathcal{P}_{\text{true}}\). Instead, we define a **pseudo-true parameter**:
\[
\theta^\circ \in \arg\min_{\theta \in \Theta} \mathrm{KL}(\mathcal{P}_{\text{true}} \Vert \mathcal{P}_\theta),
\]
where \(\mathcal{P}_\theta\) is the distribution of \(Y_{1:\infty}\) under \(\theta\).

Under general conditions, \(\hat{\theta}_T\) converges almost surely to \(\theta^\circ\). Thus, the MLE approximates the best-fitting model in the Kullback–Leibler sense.

---

## 6.3 Asymptotic Normality and Fisher Information

### 6.3.1 Score Function and Information

The **score function** is
\[
U_T(\theta) = \nabla_\theta \ell_T(\theta).
\]

The **Fisher information matrix** at \(\theta\) is
\[
I_T(\theta) = -\mathbb{E}_\theta[ \nabla_\theta^2 \ell_T(\theta) ]
= \mathbb{E}_\theta[ U_T(\theta) U_T(\theta)^\top ].
\]

For large \(T\), it is natural to study **per-observation** quantities:
\[
\bar{U}_T(\theta) = \frac{1}{\sqrt{T}} U_T(\theta), \quad
\bar{I}_T(\theta) = \frac{1}{T} I_T(\theta).
\]

Under stationarity and ergodicity, one can show that
\[
\bar{I}_T(\theta^*) \xrightarrow[T\to\infty]{} I(\theta^*),
\]
where \(I(\theta^*)\) is the **limiting Fisher information per time step**.

### 6.3.2 Central Limit Theorem for the Score

Under appropriate **mixing conditions** (e.g. geometric \(\beta\)-mixing) for the observed process \((Y_t)\), the normalized score satisfies a **central limit theorem**:
\[
\bar{U}_T(\theta^*) = \frac{1}{\sqrt{T}} \nabla_\theta \ell_T(\theta^*)
\xrightarrow{d} \mathcal{N}(0, I(\theta^*)).
\]

The proof typically relies on:

- Writing \(U_T(\theta^*)\) as a sum of a **stationary, martingale difference** sequence plus negligible terms;
- Applying a martingale CLT or a mixing CLT.

### 6.3.3 Asymptotic Normality of the MLE

Assuming:

- \(\hat{\theta}_T \to \theta^*\) almost surely (consistency);
- \(I(\theta^*)\) is **non-singular**;
- Regularity conditions for Taylor expansions;

we expand the score around \(\theta^*\):
\[
0 = U_T(\hat{\theta}_T)
= U_T(\theta^*) + \nabla_\theta^2 \ell_T(\tilde{\theta}_T) (\hat{\theta}_T - \theta^*),
\]
for some \(\tilde{\theta}_T\) between \(\hat{\theta}_T\) and \(\theta^*\).

Divide by \(\sqrt{T}\):
\[
0 = \bar{U}_T(\theta^*) + \Bigl( \frac{1}{T} \nabla_\theta^2 \ell_T(\tilde{\theta}_T) \Bigr) \sqrt{T} (\hat{\theta}_T - \theta^*).
\]

As \(T\to\infty\), the second factor converges to \(-I(\theta^*)\), and \(\bar{U}_T(\theta^*)\) converges in distribution to \(\mathcal{N}(0, I(\theta^*))\). Hence
\[
\sqrt{T}(\hat{\theta}_T - \theta^*)
\xrightarrow{d} \mathcal{N}(0, I(\theta^*)^{-1}).
\]

This is the **asymptotic normality** of the MLE.

### 6.3.4 Computing the Information in HMMs

In HMMs, \(I(\theta^*)\) can be computed using **forward–backward quantities** and expectations under the stationary distribution.

One approach:

- Express the score as
  \[
  U_T(\theta) = \sum_{t=1}^T u_t(\theta),
  \]
  where \(u_t(\theta)\) depends on local conditional distributions (e.g. \(p_\theta(S_t,S_{t+1} \mid Y_{1:T})\));
- Compute \(\mathbb{E}_{\theta^*}[u_t(\theta^*) u_s(\theta^*)^\top]\) and sum over lags.

Douc, Moulines, Stoffer provide explicit formulas and practical approximations.

---

## 6.4 Model Selection and Information Criteria

Given a family of HMMs with different numbers of states \(K\), we may select \(K\) using **information criteria** such as AIC or BIC.

The **Bayesian Information Criterion (BIC)** is
\[
\mathrm{BIC} = -2 \ell_T(\hat{\theta}_T) + d \log T,
\]
where \(d\) is the number of free parameters in \(\theta\).

Under regularity conditions, BIC is an approximation to **\(-2\) times the log marginal likelihood** (integrated over a prior), and tends to favor the **true model order** when it is among the candidates.

In HMMs, some regularity assumptions may fail (e.g. at parameter boundaries), but BIC is widely used and discussed by Zucchini et al. as a practical guide.

---

## 6.5 Summary

We have sketched the main elements of **asymptotic theory** for HMMs:

- Existence of a limiting average log-likelihood \(\ell_\infty(\theta)\) under ergodicity;
- **Consistency** of MLEs under identifiability and regularity;
- **Asymptotic normality** with covariance given by the inverse **Fisher information**;
- Behavior under **misspecification**, leading to pseudo-true parameters.

These results justify the use of MLE and information criteria in large-sample regimes, and they underpin more advanced methods such as **online estimation** and **sequential Monte Carlo** for HMMs.

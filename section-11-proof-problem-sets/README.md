# Section 11 – Proof-Based Problem Sets for HMMs

This section provides **proof-oriented exercises** designed to consolidate a rigorous understanding of HMMs. Problems range from foundational probability to advanced asymptotic theory.

They are grouped by topic; many are inspired by or extend derivations in **Zucchini et al.**, **Cappé, Moulines, Rydén**, and **Douc, Moulines, Stoffer**.

No solutions are included here; these are intended for coursework, qualifying exams, or self-study at a graduate/PhD level.

---

## 11.1 Probability and Markov Chains

1. **Sigma-algebras and conditional expectations.**  
   Let \((\Omega,\mathcal{F},\mathbb{P})\) be a probability space and \(X\) an integrable random variable. Show that the conditional expectation \(\mathbb{E}[X\mid\mathcal{G}]\) with respect to a sub-\(\sigma\)-algebra \(\mathcal{G}\subseteq\mathcal{F}\) is unique up to almost sure equality. Prove the tower property.

2. **Ergodic theorem for finite Markov chains.**  
   Let \((S_t)\) be an irreducible, aperiodic Markov chain on a finite state space with stationary distribution \(\boldsymbol{\pi}\). Prove that for any bounded function \(f\),
   \[
   \frac{1}{T} \sum_{t=1}^T f(S_t) \xrightarrow{\text{a.s.}} \sum_i \pi_i f(i).
   \]
   (Hint: use coupling or spectral methods.)

3. **Spectral gap and mixing.**  
   For a reversible Markov chain, prove that the total variation distance between \(\mathbb{P}(S_t \in \cdot \mid S_0=i)\) and \(\boldsymbol{\pi}\) decays at least geometrically with rate determined by the spectral gap \(\gamma = 1-\lambda_2\).

---

## 11.2 Inference Algorithms

4. **Forward algorithm correctness.**  
   Starting from the HMM factorization, prove by induction that the forward recursion computes \(\tilde{\alpha}_t(i) = \mathbb{P}(S_t=i,Y_{1:t}=y_{1:t})\).

5. **Forward–backward and smoothing.**  
   Derive the backward recursion and show that the smoothing probabilities satisfy
   \[
   \gamma_t(i) = \frac{\tilde{\alpha}_t(i) \beta_t(i)}{\sum_j \tilde{\alpha}_T(j)}.
   \]

6. **Viterbi optimality.**  
   Prove rigorously that the Viterbi path is a maximizer of the joint probability \(\mathbb{P}(S_{1:T},Y_{1:T})\) by showing that the dynamic programming recursion satisfies the Bellman optimality principle.

7. **Comparison of path and marginal modes.**  
   Construct an explicit example of a 2-state HMM and a short observation sequence where the sequence of marginally most probable states differs from the Viterbi path.

---

## 11.3 EM, MLE, and Identifiability

8. **EM monotonicity.**  
   Show that the EM update step satisfies
   \[
   \ell(\theta^{(k+1)}) \ge \ell(\theta^{(k)}),
   \]
   by expressing the log-likelihood as the sum of an ELBO and a KL divergence (Section 5.2.4).

9. **Complete-data sufficient statistics.**  
   For a finite-state HMM with discrete emissions, identify the complete-data sufficient statistics for \(\boldsymbol{\delta}\), \(\boldsymbol{\Gamma}\), and emission probabilities. Derive EM update formulas starting from the exponential-family structure.

10. **Label switching.**  
    Prove that permuting state labels in an HMM (and correspondingly permuting rows/columns of \(\boldsymbol{\Gamma}\) and emission parameters) yields the same distribution for \(Y_{1:T}\). Show that this is the only symmetry for generic parameter values.

11. **Non-identifiability example.**  
    Construct a simple 2-state HMM with emission distributions and transition matrix such that two distinct parameter values (not related by permutation) induce the same distribution over \(Y_{1:T}\) for all \(T\).

---

## 11.4 Asymptotics and Information

12. **Existence of limiting log-likelihood.**  
    For a stationary ergodic HMM, show (under suitable conditions) that \(\bar{\ell}_T(\theta) = T^{-1}\ell_T(\theta)\) converges almost surely to a limit \(\ell_\infty(\theta)\) for each fixed \(\theta\).

13. **Consistency of MLE.**  
    Outline a proof that \(\hat{\theta}_T\) converges to the true parameter (up to permutation) by showing that \(\ell_\infty(\theta)\) is uniquely maximized at \(\theta^*\) and using uniform convergence of \(\bar{\ell}_T\) to \(\ell_\infty\).

14. **Asymptotic normality.**  
    Derive the asymptotic distribution of \(\sqrt{T}(\hat{\theta}_T - \theta^*)\) by applying a Taylor expansion to the score and invoking a central limit theorem for \(U_T(\theta^*)\).

---

## 11.5 Advanced and Alternative Perspectives

15. **Kalman filter as linear-Gaussian HMM.**  
    Show that the Kalman filter recursion can be derived as the solution to the filtering problem in a linear-Gaussian state-space model, and compare it formally to the discrete-state forward algorithm.

16. **Nonparametric HMM identifiability (sketch).**  
    Discuss conditions under which a nonparametric HMM with infinitely many states may still be identifiable from data (e.g. via finite-rank assumptions on certain operator kernels).

17. **POMDP belief MDP.**  
    For a finite-state POMDP, prove that the process of belief states \(b_t\) forms a Markov decision process on the simplex, and write down the Bellman equations.

18. **Regret bounds for HMM predictors (conceptual).**  
    Consider the class of HMM predictors under log-loss. Formulate the notion of regret against the best fixed HMM in hindsight and outline how a Bayesian mixture or online algorithm can achieve sublinear regret.

---

## 11.6 Using These Problems

These problems are intended to be used alongside the main sections:

- 1–3 pair naturally with **Sections 0–1** (foundations and Markov chains);
- 4–7 with **Section 4** (inference algorithms);
- 8–11 with **Sections 5–6** (EM, identifiability, asymptotics);
- 15–18 with **Sections 7–9** (advanced models and alternative foundations).

Instructors can tailor subsets of these problems to build a full **graduate-level HMM course**, with Zucchini et al. as the primary applied reference and Cappé, Moulines, Rydén and Douc, Moulines, Stoffer providing the theoretical backbone.

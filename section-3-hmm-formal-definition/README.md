# Section 3 – Hidden Markov Models: Formal Definition and Likelihood

We now give a **fully formal definition** of finite-state Hidden Markov Models (HMMs) and derive the **joint** and **marginal (observed)** likelihoods.

This section closely follows the notation of **Zucchini, MacDonald, Langrock**, while making all probabilistic assumptions explicit and preparing the ground for algorithmic and statistical analysis in later sections.

---

## 3.1 Generative Definition of a Finite-State HMM

### 3.1.1 Components of the Model

Fix:

- A finite **state space** \(E = \{1,\dots,K\}\);
- An **observation space** \((\mathcal{Y}, \mathcal{B}_{\mathcal{Y}})\), e.g. \(\mathbb{R}^d\) with the Borel \(\sigma\)-algebra;
- An **initial distribution** \(\boldsymbol{\delta} = (\delta_i)_{i=1}^K\), a probability vector on \(E\);
- A **transition matrix** \(\boldsymbol{\Gamma} = (\gamma_{ij})_{i,j=1}^K\) with
  \[
  \gamma_{ij} = \mathbb{P}(S_{t+1}=j \mid S_t=i), \quad \sum_j \gamma_{ij} = 1;
  \]
- A collection of **emission distributions** \(\{F_i : i \in E\}\) on \((\mathcal{Y}, \mathcal{B}_{\mathcal{Y}})\), with densities \(f_i\) (with respect to a common dominating measure, often Lebesgue or counting measure).

### 3.1.2 Hidden State Process

On a probability space \((\Omega, \mathcal{F}, \mathbb{P})\), define a stochastic process \((S_t)_{t\ge1}\) with values in \(E\) such that

- \(\mathbb{P}(S_1 = i) = \delta_i\);
- For all \(t \ge 1\),
  \[
  \mathbb{P}(S_{t+1} = j \mid S_1,\dots,S_t) = \mathbb{P}(S_{t+1} = j \mid S_t) = \gamma_{S_t j}.
  \]

Thus, \((S_t)\) is a **time-homogeneous finite-state Markov chain** as in Section 1.

### 3.1.3 Observation Process

Given the hidden process \((S_t)\), define an observation process \((Y_t)_{t\ge1}\) taking values in \(\mathcal{Y}\) such that

- Conditional on \(S_t = i\), \(Y_t\) is drawn from \(F_i\) with density \(f_i\);
- Conditional on **all states**, observations are independent across time:
  \[
  \mathbb{P}(Y_{1:T} \in A_{1:T} \mid S_{1:T} = s_{1:T})
  = \prod_{t=1}^T F_{s_t}(A_t).
  \]

Equivalently, with densities,
\[
\mathbb{P}(Y_{1:T} \in dy_{1:T} \mid S_{1:T} = s_{1:T})
= \prod_{t=1}^T f_{s_t}(y_t) \, dy_t.
\]

The pair \((S_t, Y_t)\) defines the **Hidden Markov Model**.

---

## 3.2 Joint Likelihood Factorization

Fix a time horizon \(T\). For a realizations \(s_{1:T} \in E^T\) and \(y_{1:T} \in \mathcal{Y}^T\), the joint density (or mass function) of \((S_{1:T}, Y_{1:T})\) is
\[
\begin{aligned}
&\mathbb{P}(S_{1:T}=s_{1:T}, Y_{1:T}=y_{1:T}) \\
&= \mathbb{P}(S_1=s_1) \, \mathbb{P}(Y_1=y_1 \mid S_1=s_1)
   \prod_{t=2}^T \mathbb{P}(S_t=s_t \mid S_{t-1}=s_{t-1}) \, \mathbb{P}(Y_t=y_t \mid S_t=s_t) \\
&= \delta_{s_1} f_{s_1}(y_1) \prod_{t=2}^T \gamma_{s_{t-1}, s_t} \, f_{s_t}(y_t).
\end{aligned}
\]

This is the fundamental factorization used throughout Zucchini et al. It mirrors Equation (2.1) in their book (up to notation differences).

The **complete-data log-likelihood** (if we knew the states) is
\[
\log L_c(\boldsymbol{\delta}, \boldsymbol{\Gamma}, f; s_{1:T}, y_{1:T})
= \log \delta_{s_1} + \sum_{t=2}^T \log \gamma_{s_{t-1}, s_t}
  + \sum_{t=1}^T \log f_{s_t}(y_t).
\]

This form is crucial for the **EM/Baum–Welch algorithm** (Section 5.2).

---

## 3.3 Marginal Likelihood of the Observations

In practice, the states \(S_{1:T}\) are unobserved. The **observed data likelihood** is the marginal of the joint distribution over all possible state sequences:
\[
L(\boldsymbol{\delta}, \boldsymbol{\Gamma}, f; y_{1:T})
= \mathbb{P}(Y_{1:T}=y_{1:T})
= \sum_{s_{1:T} \in E^T} \mathbb{P}(S_{1:T}=s_{1:T}, Y_{1:T}=y_{1:T}).
\]

Substituting the joint factorization,
\[
L(\theta; y_{1:T})
= \sum_{s_{1:T}} \delta_{s_1} f_{s_1}(y_1)
  \prod_{t=2}^T \gamma_{s_{t-1}, s_t} f_{s_t}(y_t),
\]
where \(\theta\) denotes the collection of all parameters.

### 3.3.1 Naïve Computation is Exponential

There are **\(K^T\) terms** in the sum over state sequences. Direct evaluation is computationally infeasible even for moderate \(T\) and \(K\).

Example: with \(K=5\) states and \(T=100\), \(5^{100}\) is astronomically large.

Thus, we need to exploit the **Markov and conditional independence structure** to compute this marginal efficiently. This leads to the **forward algorithm** (Section 4.1), which runs in \(\mathcal{O}(K^2 T)\) time.

### 3.3.2 Matrix-Product Representation (Zucchini’s Notation)

Zucchini et al. express the likelihood using **matrix products**. Define

- A diagonal matrix of emission densities at time \(t\):
  \[
  \mathbf{Q}(y_t) = \operatorname{diag}(f_1(y_t), \dots, f_K(y_t)).
  \]

Then one can show that
\[
L(\theta; y_{1:T})
= \boldsymbol{\delta}^\top \mathbf{Q}(y_1) \boldsymbol{\Gamma} \mathbf{Q}(y_2) \cdots \boldsymbol{\Gamma} \mathbf{Q}(y_T) \mathbf{1},
\]
where \(\mathbf{1}\) is the column vector of ones.

**Derivation (sketch):** each matrix multiplication corresponds to summing over an intermediate state index. The product 
\(\mathbf{Q}(y_t)\boldsymbol{\Gamma}\mathbf{Q}(y_{t+1})\) encodes the contribution of transitions from time \(t\) to \(t+1\) and emissions at both times.

This matrix formulation is central in Zucchini et al. and will match the **forward variable recursion** in Section 4.

---

## 3.4 Log-Likelihood and Its Geometry

The **log-likelihood** is
\[
\ell(\theta; y_{1:T}) = \log L(\theta; y_{1:T}).
\]

Properties:

- \(\ell\) is typically **non-convex** in \(\theta\) due to hidden states and combinatorial symmetries (label switching);
- It is, however, **smooth** in the interior of the parameter space (for regular emission families);
- Gradient and Hessian can be expressed in terms of **forward–backward quantities** and **conditional expectations**.

These observations motivate the **EM algorithm**: instead of maximizing \(\ell\) directly, one maximizes a **lower bound** (Section 5.2), whose geometry is often easier.

---

## 3.5 Parameter Space and Constraints

The parameter space naturally decomposes as
\[
\Theta = \Delta^{K-1} \times \mathcal{G} \times \Phi,
\]
where

- \(\Delta^{K-1}\) is the simplex for the initial distribution \(\boldsymbol{\delta}\);
- \(\mathcal{G}\) is the set of \(K\times K\) row-stochastic matrices \(\boldsymbol{\Gamma}\);
- \(\Phi\) is the product of emission parameter spaces \(\Phi_1 \times \cdots \times \Phi_K\).

Constraints:

- \(\delta_i \ge 0, \sum_i \delta_i = 1\);
- \(\gamma_{ij} \ge 0, \sum_j \gamma_{ij} = 1\) for each \(i\);
- Emission parameters must keep \(f_i\) valid probability distributions.

Optimization (MLE, EM) must respect these constraints; many algorithms use **reparameterizations** (e.g. softmax/logistic transforms) to enforce them automatically.

---

## 3.6 Summary

In this section we:

- Formally defined a finite-state HMM as a pair of processes \((S_t, Y_t)\) with a Markov hidden chain and conditionally independent emissions;
- Derived the **joint** likelihood of states and observations;
- Obtained the **marginal** likelihood as a sum over \(K^T\) state sequences;
- Introduced the **matrix-product representation** of the likelihood used extensively by Zucchini et al.

This sets the stage for:

- **Section 4:** Efficient inference algorithms (forward–backward, Viterbi) that compute various conditional probabilities and the likelihood in \(\mathcal{O}(K^2 T)\);
- **Section 5:** Parameter estimation via maximum likelihood and EM/Baum–Welch.

For a detailed treatment closely aligned with this notation, see Zucchini et al., **Chapter 2 (The HMM)**.

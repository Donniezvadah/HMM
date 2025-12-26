# Section 4 – Inference in Hidden Markov Models

This section develops the **core inference algorithms** for finite-state HMMs:

- **Filtering (forward algorithm)** – computing $\mathbb{P}(S_t = i \mid Y_{1:t})$;
- **Smoothing (forward–backward)** – computing $\mathbb{P}(S_t = i \mid Y_{1:T})$;
- **Decoding (Viterbi)** – computing the most probable state sequence $\arg\max_{s_{1:T}} \mathbb{P}(S_{1:T}=s_{1:T} \mid Y_{1:T})$.

We emphasize **recursive structure**, **dynamic programming**, **proofs of correctness**, and **numerical stability**.

The treatment aligns with **Zucchini et al.**, Chapters 2–3, and **Rabiner (1989)**, but is more explicit about the probabilistic underpinnings.

Throughout, $\theta = (\boldsymbol{\delta}, \boldsymbol{\Gamma}, f_1,\dots,f_K)$ denotes the HMM parameters, and we condition implicitly on $\theta$ when unambiguous.

---

## 4.1 Filtering – The Forward Algorithm

### 4.1.1 Filtering and Predictive Distributions

Given observations $Y_{1:t} = y_{1:t}$, define

- The **filtering distribution** (posterior over states):
  $$
  \alpha_t(i) := \mathbb{P}(S_t = i \mid Y_{1:t} = y_{1:t}), \quad i=1,\dots,K.
  $$
- The **one-step predictive distribution**:
  $$
  \mathbb{P}(Y_{t+1} \in A \mid Y_{1:t} = y_{1:t})
  = \sum_{i=1}^K \mathbb{P}(S_t = i \mid Y_{1:t})
    \sum_{j=1}^K \gamma_{ij} F_j(A).
  $$

The forward algorithm computes all $\alpha_t$ **recursively in $t$**, in $\mathcal{O}(K^2 T)$ time.

### 4.1.2 Unnormalized Forward Variables

Define the **unnormalized forward variables**
$$
\tilde{\alpha}_t(i) := \mathbb{P}(S_t = i, Y_{1:t} = y_{1:t}).
$$

Then
$$
\alpha_t(i) = \frac{\tilde{\alpha}_t(i)}{\sum_{j=1}^K \tilde{\alpha}_t(j)}.
$$

The forward recursion is most naturally stated for $\tilde{\alpha}_t(i)$.

### 4.1.3 Derivation of the Recursion

**Initialization (t = 1).**
$$
\tilde{\alpha}_1(i) = \mathbb{P}(S_1 = i, Y_1 = y_1)
= \mathbb{P}(S_1 = i) \, \mathbb{P}(Y_1 = y_1 \mid S_1 = i)
= \delta_i f_i(y_1).
$$

**Induction step.** For $t \ge 1$,
$$
\begin{aligned}
\tilde{\alpha}_{t+1}(j)
&= \mathbb{P}(S_{t+1} = j, Y_{1:t+1} = y_{1:t+1}) \\
&= \sum_{i=1}^K \mathbb{P}(S_t = i, S_{t+1} = j, Y_{1:t+1} = y_{1:t+1}) \\
&= \sum_{i=1}^K \mathbb{P}(S_t = i, Y_{1:t} = y_{1:t}) \\
&\quad   \mathbb{P}(S_{t+1} = j \mid S_t = i) \\
&\quad   \mathbb{P}(Y_{t+1} = y_{t+1} \mid S_{t+1} = j) \\
&= \sum_{i=1}^K \tilde{\alpha}_t(i) \, \gamma_{ij} \, f_j(y_{t+1}).
\end{aligned}
$$

The key step uses:

- The **Markov property** for $S_t$;
- Conditional independence of $Y_{t+1}$ from the past given $S_{t+1}$.

Thus the recursion is
$$
\boxed{\tilde{\alpha}_{t+1}(j) = f_j(y_{t+1}) \sum_{i=1}^K \tilde{\alpha}_t(i) \, \gamma_{ij}.}
$$

### 4.1.4 Matrix Formulation (Zucchini’s Notation)

Let

- $\boldsymbol{\tilde{\alpha}}_t$ be the row vector with entries $\tilde{\alpha}_t(i)$;
- $\mathbf{Q}(y_t) = \operatorname{diag}(f_1(y_t),\dots,f_K(y_t))$ as before.

Then
$$
\boldsymbol{\tilde{\alpha}}_1 = \boldsymbol{\delta}^\top \mathbf{Q}(y_1),
$$
$$
\boldsymbol{\tilde{\alpha}}_{t+1} = \boldsymbol{\tilde{\alpha}}_t \, \boldsymbol{\Gamma} \, \mathbf{Q}(y_{t+1}).
$$

This matches precisely the likelihood expression in Section 3.3: the marginal likelihood is
$$
L(\theta; y_{1:T}) = \sum_{i=1}^K \tilde{\alpha}_T(i) = \boldsymbol{\tilde{\alpha}}_T \mathbf{1}.
$$

Zucchini et al. use this matrix-product viewpoint extensively; the forward algorithm is exactly this recursion plus normalization at each step.

### 4.1.5 Proof of Correctness by Induction

We show that the recursion indeed computes $\tilde{\alpha}_t(i) = \mathbb{P}(S_t=i, Y_{1:t}=y_{1:t})$ for all $t$.

- **Base case:** Already verified for $t=1$.
- **Induction step:** Assume formula holds for $t$. Then using only the model assumptions (Markov property and conditional independence), we derived the recursion, which equals by definition
  $$
  \mathbb{P}(S_{t+1} = j, Y_{1:t+1} = y_{1:t+1}).
  $$

Hence, by induction, the recursion is correct for all $t$. This is the standard argument also given in Zucchini et al. (with lighter measure-theoretic detail).

### 4.1.6 Numerical Stability: Scaling and Log-Domain

Direct computation of $\tilde{\alpha}_t(i)$ leads to **underflow**, since they involve products of $T$ probabilities. Two standard cures:

1. **Scaling:** At each step define a scaling constant
   $$
   c_t = \sum_{i=1}^K \tilde{\alpha}_t(i), \quad \hat{\alpha}_t(i) = \frac{\tilde{\alpha}_t(i)}{c_t}.
   $$
   Then $\hat{\alpha}_t$ is the normalized filtering distribution, and
   $$
   L(\theta; y_{1:T}) = \prod_{t=1}^T c_t, \quad \ell(\theta; y_{1:T}) = \sum_{t=1}^T \log c_t.
   $$
   This is exactly the implementation recommended in Zucchini et al.

2. **Log-domain forward algorithm:** Work with
   $$
   a_t(i) = \log \tilde{\alpha}_t(i),
   $$
   and use the **log-sum-exp** trick for the recursion:
   $$
   a_{t+1}(j) = \log f_j(y_{t+1}) + \log \Bigl( \sum_{i=1}^K e^{a_t(i) + \log \gamma_{ij}} \Bigr).
   $$
   Numerically, compute
   $$
   \log \sum_{i} e^{z_i} = m + \log \sum_i e^{z_i - m}, \quad m = \max_i z_i,
   $$
   to avoid overflow and underflow.

---

## 4.2 Smoothing – Forward–Backward Algorithm

Filtering uses observations up to time $t$. For many tasks (e.g. EM, state decoding), we need **smoothing distributions** that use the **entire sequence** $Y_{1:T}$.

### 4.2.1 Smoothing Distributions and Backward Variables

Define the **smoothing distribution** at time $t$:
$$
\gamma_t(i) := \mathbb{P}(S_t = i \mid Y_{1:T} = y_{1:T}).
$$

Introduce **backward variables**
$$
\beta_t(i) := \mathbb{P}(Y_{t+1:T} = y_{t+1:T} \mid S_t = i).
$$

Intuitively, $\beta_t(i)$ is the probability of observing the future $y_{t+1:T}$ if we know the current state is $i$.

### 4.2.2 Backward Recursion

**Initialization:** At time $T$, there are no future observations, so by convention
$$
\beta_T(i) = 1, \quad i=1,\dots,K.
$$

**Induction step:** For $t = T-1,\dots,1$,
$$
\begin{aligned}
\beta_t(i)
&= \mathbb{P}(Y_{t+1:T} = y_{t+1:T} \mid S_t = i) \\
&= \sum_{j=1}^K \mathbb{P}(S_{t+1} = j, Y_{t+1:T} = y_{t+1:T} \mid S_t = i) \\
&= \sum_{j=1}^K \gamma_{ij} f_j(y_{t+1}) \beta_{t+1}(j).
\end{aligned}
$$

Hence the **backward recursion** is
$$
\boxed{\beta_t(i) = \sum_{j=1}^K \gamma_{ij} f_j(y_{t+1}) \beta_{t+1}(j).}
$$

### 4.2.3 Two-Filter Formula: Combining Forward and Backward

We have
$$
\begin{aligned}
\mathbb{P}(S_t = i, Y_{1:T} = y_{1:T})
&= \mathbb{P}(S_t = i, Y_{1:t} = y_{1:t}) \\
&\quad   \mathbb{P}(Y_{t+1:T} = y_{t+1:T} \mid S_t = i, Y_{1:t}=y_{1:t}) \\
&= \tilde{\alpha}_t(i) \, \beta_t(i),
\end{aligned}
$$

since **future observations are conditionally independent of the past given $S_t$**.

Thus the smoothing distribution is
$$
\gamma_t(i) = \mathbb{P}(S_t = i \mid Y_{1:T} = y_{1:T})
= \frac{\tilde{\alpha}_t(i) \, \beta_t(i)}{L(\theta; y_{1:T})}.
$$

In scaled form, using $\hat{\alpha}_t(i)$ and scaled $\hat{\beta}_t(i)$, the denominator cancels nicely (see Zucchini et al. for implementation details):
$$
\gamma_t(i) \propto \hat{\alpha}_t(i) \, \hat{\beta}_t(i),
$$
with proportionality factors determined by normalization.

### 4.2.4 Pairwise Smoothing Probabilities

For EM/Baum–Welch, we also need
$$
\xi_t(i,j) := \mathbb{P}(S_t=i, S_{t+1}=j \mid Y_{1:T}=y_{1:T}).
$$

Using similar reasoning,
$$
\xi_t(i,j) = \frac{\tilde{\alpha}_t(i) \, \gamma_{ij} f_j(y_{t+1}) \beta_{t+1}(j)}{L(\theta; y_{1:T})}.
$$

The arrays $\gamma_t(i)$ and $\xi_t(i,j)$ are exactly what EM uses as **expected sufficient statistics** for state occupancies and transitions.

---

## 4.3 Decoding – The Viterbi Algorithm

Filtering and smoothing give **marginal posterior distributions** over states at each time. In many applications, one wants a **single state sequence estimate** $\hat{s}_{1:T}$.

The most common choice is the **maximum a posteriori (MAP) path**:
$$
\hat{s}_{1:T}^{\text{MAP}} \in \arg\max_{s_{1:T}} \mathbb{P}(S_{1:T}=s_{1:T} \mid Y_{1:T}=y_{1:T}).
$$

Equivalently,
$$
\hat{s}_{1:T}^{\text{MAP}} \in \arg\max_{s_{1:T}} \mathbb{P}(S_{1:T}=s_{1:T}, Y_{1:T}=y_{1:T}),
$$
since the denominator $\mathbb{P}(Y_{1:T}=y_{1:T})$ does not depend on $s_{1:T}$.

### 4.3.1 Dynamic Programming Formulation

Define
$$
\delta_t(j) := \max_{s_{1:t-1}} \mathbb{P}(S_t = j, S_{1:t-1}=s_{1:t-1}, Y_{1:t}=y_{1:t}),
$$
and the **backpointer**
$$
\psi_t(j) \in \arg\max_{i} \delta_{t-1}(i) \gamma_{ij}.
$$

Then the Viterbi recursion is:

- **Initialization:**
  $$
  \delta_1(j) = \delta_j f_j(y_1), \quad \psi_1(j) \text{ arbitrary}.
  $$
- **Recursion:** for $t=2,\dots,T$,
  $$
  \delta_t(j) = f_j(y_t) \max_{i} \delta_{t-1}(i) \gamma_{ij},
  $$
  $$
  \psi_t(j) \in \arg\max_{i} \delta_{t-1}(i) \gamma_{ij}.
  $$
- **Termination:**
  $$
  \hat{s}_T \in \arg\max_j \delta_T(j).
  $$
- **Backtracking:** For $t=T-1,\dots,1$,
  $$
  \hat{s}_t = \psi_{t+1}(\hat{s}_{t+1}).
  $$

### 4.3.2 Proof of Correctness

The Viterbi algorithm is an instance of **dynamic programming** over a chain:

- For each $t,j$, $\delta_t(j)$ is the **maximum joint probability** over all paths ending in state $j$ at time $t$;
- The optimal path to $j$ at time $t$ must pass through some $i$ at time $t-1$, and that prefix must be optimal for reaching $i$ at time $t-1$.

Formally, one proves by induction:

1. **Optimal substructure:** if $s_{1:T}^*$ maximizes $\mathbb{P}(S_{1:T},Y_{1:T})$, then for each $t$, the prefix $s_{1:t}^*$ must maximize $\mathbb{P}(S_{1:t},Y_{1:t})$ among all paths ending in $s_t^*$;
2. The recursion above computes exactly these maxima.

See Zucchini et al., Chapter 3, and Rabiner (1989) for standard textbook proofs.

### 4.3.3 Max-Product Semiring Perspective

The Viterbi algorithm can be seen as a **max-product message passing** on the chain factor graph:

- Replace summation (as in forward algorithm) by maximization;
- Replace probabilities by their **logarithms**, turning products into sums:
  $$
  v_t(j) = \log \delta_t(j)
          = \log f_j(y_t) + \max_i \{ v_{t-1}(i) + \log \gamma_{ij} \} + \log \delta_j \mathbf{1}_{t=1}.
  $$

This semiring viewpoint is useful when generalizing to other objectives (e.g. **min-sum** for costs).

### 4.3.4 Complexity and Path Properties

- Time complexity is $\mathcal{O}(K^2 T)$, same order as forward–backward;
- Memory complexity is $\mathcal{O}(K T)$ if all $\psi_t(j)$ are stored; can be reduced with more complex techniques.

Importantly, the **Viterbi path is not obtained by taking the most likely state at each time** (that would use $\gamma_t(i)$), because the most likely joint path is not obtained by locally maximizing each marginal.

---

## 4.4 Other Inference Quantities

From filtering and smoothing, one can derive many other useful quantities:

- **Predictive distribution:**
  $$
  \mathbb{P}(Y_{t+1} \in A \mid Y_{1:t})
  = \sum_{i,j} \alpha_t(i) \gamma_{ij} F_j(A).
  $$
- **State occupancy expectations:** $\mathbb{E}[\mathbf{1}_{\{S_t=i\}} \mid Y_{1:T}] = \gamma_t(i)$.
- **Expected transition counts:** $\mathbb{E}[\mathbf{1}_{\{S_t=i,S_{t+1}=j\}} \mid Y_{1:T}] = \xi_t(i,j)$.

These are central to **parameter estimation** (Section 5) and to interpreting HMMs in applications (Section 10).

---

## 4.5 Summary and References

We have developed:

- The **forward algorithm** for filtering, with rigorous derivation and scaling for numerical stability;
- The **backward recursion** and the forward–backward method for **smoothing** and pairwise probabilities;
- The **Viterbi algorithm** for MAP path decoding, with a dynamic programming interpretation and proof sketch.

These algorithms are the computational workhorses of HMM inference. Zucchini et al., **Chapters 2–3**, provide code-oriented explanations (often in R), while the more formal treatment here is aligned with **Cappé, Moulines, Rydén (2005)** and **Rabiner (1989)**.

# Section 9 – Alternative Foundations for HMMs

This section explores **non-standard perspectives** on HMMs that go beyond classical likelihood-based estimation:

- **Online and distribution-free viewpoints**, including prediction with expert advice and regret bounds;
- **Decision-theoretic framing** of HMMs as partially observable control problems (POMDPs).

These perspectives are not central in Zucchini et al., but are powerful for understanding HMMs in **sequential decision-making** and **adversarial or non-stationary environments**.

---

## 9.1 Online Prediction and Regret

### 9.1.1 Prediction Problem Setup

Consider a sequence of observations $Y_1, Y_2, \dots$ taking values in a measurable space $\mathcal{Y}$. At each time $t$:

1. The forecaster outputs a predictive distribution $q_t$ over $Y_t$ based on $Y_{1:t-1}$;
2. The true outcome $Y_t$ is revealed;
3. The forecaster incurs a loss $\ell(q_t, Y_t)$, often **log-loss**:
   $$
   \ell(q_t, Y_t) = -\log q_t(Y_t).
   $$

An HMM with parameter $\theta$ induces a natural predictive distribution
$$
 q_t^\theta(\cdot) = p_\theta(\cdot \mid Y_{1:t-1}).
$$

The question: how do such predictors perform in an **online** or **adversarial** setting?

### 9.1.2 Regret Against a Class of HMMs

Fix a class of HMMs $\{p_\theta : \theta \in \Theta\}$. The **cumulative log-loss** of predictor $q$ up to time $T$ is
$$
L_T(q) = \sum_{t=1}^T -\log q_t(Y_t).
$$

The **regret** against the best HMM in hindsight is
$$
R_T(q) = L_T(q) - \inf_{\theta \in \Theta} L_T(q^\theta).
$$

One can design online algorithms (e.g. mixture-based or Bayesian) whose regret grows **sublinearly** in $T$, ensuring that the average additional loss **vanishes** asymptotically.

This connects to the **universal prediction** literature, where HMMs serve as a rich, structured class of experts.

### 9.1.3 Bayesian Mixture over HMMs

Consider a prior $\Pi$ over $\Theta$, and define the **Bayesian mixture predictor**
$$
q_t^{\text{mix}}(\cdot)
= \int p_\theta(\cdot \mid Y_{1:t-1}) \, \Pi(d\theta \mid Y_{1:t-1}),
$$
where $\Pi(\cdot \mid Y_{1:t-1})$ is the posterior over $\theta$.

Under log-loss, such mixture predictors achieve near-optimal regret bounds against the best $\theta$ in $\Theta$. This is an example of **distribution-free performance guarantees** — no assumptions are made on how $Y_t$ are generated.

---

## 9.2 Decision-Theoretic Framing and POMDPs

### 9.2.1 HMMs as Partially Observable Markov Decision Processes

A **Partially Observable Markov Decision Process (POMDP)** consists of:

- Hidden states $S_t$ in a set $E$;
- Actions $A_t$ in an action set $\mathcal{A}$;
- Observations $Y_t$ in $\mathcal{Y}$;
- Transition probabilities $p(s_{t+1} \mid s_t, a_t)$;
- Observation probabilities $p(y_t \mid s_t)$;
- Reward (or cost) function $r(s_t, a_t)$.

An HMM is a **degenerate POMDP** with **no actions** (or a single trivial action) and no explicit rewards. Nevertheless, framing HMMs as POMDPs is useful:

- The **belief state** $b_t(i) = \mathbb{P}(S_t=i \mid Y_{1:t})$ is a sufficient statistic for the history;
- Filtering (forward algorithm) is exactly the **belief update** in a POMDP.

### 9.2.2 Control and Decision Problems with HMMs

In many applications, we do not only wish to **infer** the hidden states but also to perform **actions** based on our beliefs:

- **Maintenance / reliability:** hidden state models system health; actions trigger inspections/repairs;
- **Finance:** hidden regimes guide trading decisions;
- **Medicine:** hidden disease states guide treatment decisions.

Formally, we want to choose policies $\pi$ mapping belief states (or observation histories) to actions, to maximize expected cumulative reward:
$$
\max_\pi $$ \mathbb{E}\Bigg[ \sum_{t=1}^T r(S_t, A_t) \Bigg].
$$

### 9.2.3 Dynamic Programming in Belief Space

In a POMDP, the optimal policy can be obtained by dynamic programming on the space of **beliefs** (probability distributions over states). For finite-state HMMs, the belief space is the simplex $\Delta^{K-1}$.

The value function $V_t(b)$ satisfies a **Bellman equation** of the form
$$
V_t(b) = \max_{a \in \mathcal{A}} \Big\{ r(b,a) + \mathbb{E}[ V_{t+1}(b') \mid b,a ] \Big\},
$$
where $b'$ is the updated belief after taking action $a$ and receiving observation $Y_{t+1}$.

The belief update is exactly the **Bayesian filtering step**, which for HMM-like POMDPs is a linear-fractional map on $\Delta^{K-1}$, followed by normalization.

### 9.2.4 Risk-Sensitive and Robust Objectives

Beyond expected reward, one can study **risk-sensitive** or **robust** criteria:

- **Exponential utility:** maximize $-\frac{1}{\lambda} \log \mathbb{E}[ e^{-\lambda \sum r_t} ]$, linking to KL-regularized control;
- **Minimax regret:** choose policies that minimize the worst-case regret relative to a class of models.

These formulations often involve **entropy** and **KL divergence**, connecting back to Section 0.3 and EM-style variational principles.

---

## 9.3 Summary

This section reframed HMMs in two broader contexts:

- As **online predictors** within a regret-minimization framework, where their performance can be compared against the best model in hindsight without assuming a true generative distribution;
- As special cases of **POMDPs**, where belief updates (filtering) are combined with **decision-making** and **control**.

These perspectives link the probabilistic foundations of HMMs (as in Zucchini et al.) with modern work in **online learning**, **reinforcement learning**, and **robust control**.

# Section 10 – Applications of Hidden Markov Models

This section sketches **major application domains** of HMMs, emphasizing **precise mathematical formulations** rather than informal stories. For each domain we describe:

- The **state space** and its interpretation;
- The **observation model** (emissions);
- The **transition structure** and its constraints;
- The **inference or decision problem** being solved.

Zucchini et al. provide many application examples (e.g. animal movement, environmental data). Here we emphasize a few canonical areas.

---

## 10.1 Speech Recognition

### 10.1.1 Model Structure

In classical **speech recognition**, an HMM is used to model the mapping from hidden linguistic units to acoustic features:

- Hidden states $S_t$: phonetic units (phones), context-dependent phones, or sub-phonetic states;
- Observations $Y_t$: short-time acoustic feature vectors (e.g. MFCCs) in $\mathbb{R}^d$;
- Transition matrix $\boldsymbol{\Gamma}$: encodes allowed transitions between phones (including self-transitions for duration modeling);
- Emission distributions $f_i(y)$: often Gaussian mixtures or more complex distributions over acoustic features.

### 10.1.2 Inference Tasks

- **Likelihood computation:** $p_\theta(Y_{1:T})$ for a given sequence of acoustic features and a candidate word sequence;
- **Decoding:** find the most likely sequence of phones or words given observations (Viterbi);
- **Training:** MLE of HMM parameters via EM/Baum–Welch, often embedded inside larger systems (e.g. with language models).

Rabiner (1989) remains a classic reference for this application, describing HMMs as the central modeling tool for early speech systems.

---

## 10.2 Bioinformatics

### 10.2.1 CpG Island Detection

In genomics, HMMs can model regions with different **nucleotide composition**, such as **CpG islands**.

- Hidden states: $S_t \in \{\text{island}, \text{non-island}\}$;
- Observations: nucleotides $Y_t \in \{\text{A},\text{C},\text{G},\text{T}\}$;
- Emissions: state-dependent multinomial distributions over nucleotides;
- Transitions: probabilities governing the length and frequency of CpG islands.

Inference tasks:

- **Decoding:** identify which positions belong to islands vs background (Viterbi or posterior decoding);
- **Parameter estimation:** learn emission probabilities and transition rates from annotated or unannotated sequences.

### 10.2.2 Sequence Alignment and Profile HMMs

**Profile HMMs** generalize simple HMMs for **multiple sequence alignment**:

- States represent positions in an alignment (match, insert, delete);
- Emissions correspond to amino acids or nucleotides;
- Transitions model gaps and alignment patterns.

While structurally more complex, they are still HMMs with specialized topology.

---

## 10.3 Finance and Econometrics

### 10.3.1 Regime-Switching Models

In finance, HMMs model **regime changes** in returns (e.g. bull vs bear markets):

- Hidden states: $S_t \in \{1,\dots,K\}$ representing regimes (e.g. low-volatility vs high-volatility);
- Observations: asset returns $Y_t \in \mathbb{R}$ or $\mathbb{R}^d$;
- Emissions: state-dependent distributions, often Gaussian with mean $\mu_i$ and variance $\sigma_i^2$ per state $i$;
- Transitions: Markov matrix encoding persistence of regimes.

The model is
$$
Y_t \mid S_t = i \sim \mathcal{N}(\mu_i, \sigma_i^2),
$$
with $(S_t)$ as in Section 1.

Inference tasks:

- **Filtering / smoothing:** posterior probabilities of regimes given returns, for risk management and forecasting;
- **Parameter estimation:** MLE via EM;
- **Regime-dependent decision-making:** portfolio allocation or hedging strategies that depend on inferred regimes.

### 10.3.2 Markov-Switching Autoregressions

More generally, one can have **Markov-switching AR models** where
$$
Y_t = \mu_{S_t} + \phi_{S_t} Y_{t-1} + \varepsilon_t,
$$
with regime-dependent AR coefficients. This is an HMM in an extended state space and is closely related to **switching state-space models** (Section 7.3).

---

## 10.4 Epidemiology and Latent Disease States

### 10.4.1 Disease Progression Models

In epidemiology and biostatistics, HMMs can model **disease progression** where the true disease state is partially observed:

- Hidden states: discrete health states (e.g. healthy, infected, recovered) or stages (e.g. early, advanced);
- Observations: noisy test results, symptoms, biomarkers;
- Transitions: disease progression probabilities influenced by covariates (e.g. age, treatment).

The HMM structure is:

- $S_t$ evolves as a Markov chain with transition matrix possibly depending on covariates;
- $Y_t$ arises from state-dependent emission distributions (e.g. logistic regression for test outcomes).

Inference tasks:

- Estimating **transition probabilities** and **state occupancy** probabilities over time;
- Designing **screening and treatment policies** based on inferred states.

---

## 10.5 General Modeling Pattern (Zucchini et al.)

Zucchini et al. emphasize a common pattern across applications:

1. **Choose a number of states** $K$ and interpret them substantively (e.g. behavior modes, regimes);
2. Specify a **state process** (transition matrix, possibly with covariates);
3. Choose **emission distributions** compatible with the data type (discrete, continuous, circular, multivariate);
4. Fit the model via **MLE/EM** and evaluate via likelihood-based criteria and diagnostics;
5. Use decoding and posterior state probabilities for **interpretation** and **decision-making**.

---

## 10.6 Summary

This section highlighted how the **abstract HMM framework** is instantiated in:

- **Speech recognition** (linguistic units $\to$ acoustic features);
- **Bioinformatics** (genomic regions, alignment profiles);
- **Finance** (market regimes and volatility states);
- **Epidemiology** (latent disease progression).

In all cases, the core mathematical machinery — **Markov chains**, **emission models**, and **inference algorithms** — is exactly that developed in Sections 1–5, as presented systematically in Zucchini et al.

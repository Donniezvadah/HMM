# Hidden Markov Models (HMMs): A Rigorous, Mathematically Heavy Course

This project is a **full, proof-oriented course on Hidden Markov Models (HMMs)**, designed at the level of a serious graduate or early PhD sequence.

The course emphasizes:

- **Probability theory and stochastic processes** (measure-theoretic where needed)
- **Inference and algorithms** (forward–backward, Viterbi, EM/Baum–Welch) with **full derivations and proofs of correctness**
- **Statistical theory** (consistency, asymptotic normality, identifiability)
- **Advanced variants** (continuous-state, nonparametric, switching models)

The materials are organized into **12 sections (0–11)**. Each section lives in its own directory, with a dedicated `README.md` containing **detailed notes, theorems, and proof sketches**.

---

## Primary References (Used for Notation and Examples)

The exposition and notation lean heavily on:

- **Zucchini, MacDonald, Langrock** – *Hidden Markov Models for Time Series: An Introduction Using R* (2nd ed.).  
  This is the **main guiding reference** for finite-state HMMs, likelihoods, algorithms, and many examples.
- **Rabiner (1989)** – *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition*.  
  Classic algorithmic exposition (forward–backward, Viterbi, Baum–Welch).
- **Cappé, Moulines, Rydén (2005)** – *Inference in Hidden Markov Models*.  
  Deep, rigorous treatment of HMM inference and statistical properties.
- **Douc, Moulines, Stoffer (2014)** – *Nonlinear Time Series: Theory, Methods and Applications*.  
  Asymptotic theory and ergodic properties for dependent data, including HMMs.
- **Murphy (2012)** – *Machine Learning: A Probabilistic Perspective*.  
  Broad probabilistic graphical model framing.

Unless otherwise noted, **notation follows Zucchini et al.** where feasible:

- Hidden state process: $(S_t)_{t\ge 1}$, taking values in a finite set $\{1,\dots,K\}$
- Observation process: $(Y_t)_{t\ge 1}$
- Initial distribution: $\boldsymbol{\delta} = (\delta_i)_{i=1}^K$
- Transition probability matrix: $\boldsymbol{\Gamma} = (\gamma_{ij})_{i,j=1}^K$
- State-dependent (emission) densities or pmfs: $f_i(\cdot)$ for state $i$

---

## Course Structure (Section Index)

Each bullet links to a folder containing a **section-specific `README.md`**.

- **[0. Mathematical Prerequisites](section-0-mathematical-prerequisites/README.md)**  
  Measure-theoretic probability (light but precise), linear algebra and spectral theory for stochastic matrices, convexity and information geometry (KL divergence as a Bregman divergence).

- **[1. Markov Chains (Fully Rigorous)](section-1-markov-chains/README.md)**  
  Finite-state Markov chains, Chapman–Kolmogorov equations, stationary and invariant distributions, reversibility, ergodic theory (irreducibility, aperiodicity, mixing times, spectral gaps), and non-homogeneous chains.

- **[2. Observation Models and Emission Processes](section-2-observation-models/README.md)**  
  Graphical model formulation of HMMs, conditional independence structure, factorization of joint distributions, discrete/continuous/exponential-family emissions, and identifiability issues.

- **[3. Hidden Markov Models: Formal Definition](section-3-hmm-formal-definition/README.md)**  
  Generative definition of HMMs, formal state and observation spaces, initial distribution, transition kernel, emission kernel, and rigorous derivation of the joint and marginal likelihood.

- **[4. Inference in HMMs (Core Algorithms)](section-4-inference/README.md)**  
  Filtering (forward algorithm), smoothing (forward–backward), and decoding (Viterbi). Includes dynamic programming derivations, correctness proofs, and numerical stability considerations.

- **[5. Parameter Estimation](section-5-parameter-estimation/README.md)**  
  Maximum likelihood estimation, EM/Baum–Welch algorithm (as coordinate ascent on an evidence lower bound), monotonicity and convergence guarantees, and identifiability theory.

- **[6. Asymptotics and Statistical Theory](section-6-asymptotics/README.md)**  
  Consistency and asymptotic normality of MLE in ergodic HMMs, pseudo-true parameters under misspecification, Fisher information for dependent data.

- **[7. Non-Standard and Advanced HMMs](section-7-advanced-hmms/README.md)**  
  Continuous-state HMMs (including linear Gaussian / Kalman models), nonparametric HMMs (e.g. Dirichlet process HMMs), and switching state-space models.

- **[8. Computational and Numerical Issues](section-8-computational-issues/README.md)**  
  Scaling and log-domain implementations, underflow and overflow analysis, complexity of exact inference (time and space), and approximate methods.

- **[9. Alternative Foundations](section-9-alternative-foundations/README.md)**  
  Online and distribution-free perspectives, prediction with expert-advice style losses, regret bounds for HMM-like models, decision-theoretic framing via POMDPs.

- **[10. Applications](section-10-applications/README.md)**  
  Full mathematical mapping of real applications: speech recognition, bioinformatics, finance, epidemiology, and more, always phrased as precise HMMs.

- **[11. Proof-Based Problem Sets](section-11-proof-problem-sets/README.md)**  
  Collections of theorem-level exercises: proving algorithm correctness, constructing counterexamples, identifiability and stability proofs, and asymptotic bounds.

---

## How to Use This Course

- **Read Sections 0–1 carefully** if your background in probability or Markov chains is not fully measure-theoretic.  
- **Work through the proofs** in Sections 3–5; they are central to a deep understanding of HMMs. Zucchini et al. provide many of the key derivations, which are expanded here.
- **Use Sections 6–9** as advanced material or for a second pass when you care about asymptotics, nonparametric models, or decision-theoretic views.
- **Attempt the problem sets in Section 11** as if they were exam or qualifying questions.

Roughly:

- **70%** of the course is probability and inference theory
- **20%** is algorithms with correctness proofs
- **10%** is applications and modeling case studies

---

## Deployment and Final Site Build

To build and deploy the course website (a Quarto book with output in `_site/`):

- **1. Render the full site locally**

  ```bash
  quarto render
  ```

  This generates the static HTML site into the `_site/` directory as configured in `_quarto.yml`.

- **2. Preview locally (optional)**

  ```bash
  quarto preview
  ```

  This starts a local web server so you can inspect the site before publishing.

- **3. Deploy to GitHub Pages (recommended if using GitHub)**

  From the project root:

  ```bash
  quarto publish gh-pages
  ```

  This will:

  - Build the site
  - Push the rendered `_site/` contents to the `gh-pages` branch
  - Configure it for GitHub Pages hosting

- **4. Deploy to any static host (Netlify, Vercel, custom server)**

  - Configure your host to use the project root as the build directory
  - Set the **build command** to:

    ```bash
    quarto render
    ```

  - Set the **publish directory** (or equivalent) to:

    ```
    _site
    ```

  Any static host that can serve a folder of HTML/JS/CSS files can use the contents of `_site/` as the final deployed site.

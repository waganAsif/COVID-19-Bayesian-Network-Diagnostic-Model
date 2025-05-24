# Contributing to COVID-19 Bayesian Network Diagnostic Model

Thank you for your interest in contributing to the **COVID-19 Bayesian Network Diagnostic Model**!
This project applies **Bayesian inference**, **Dirichlet priors**, and the **BDeu scoring metric** to enable accurate diagnostics for COVID-19, especially under data uncertainty. We welcome contributions from developers, data scientists, clinicians, and researchers.

---

## 📂 Table of Contents

* [Code of Conduct](#code-of-conduct)
* [Getting Started](#getting-started)
* [How to Contribute](#how-to-contribute)
* [Development Setup](#development-setup)
* [Contribution Guidelines](#contribution-guidelines)
* [Model Development Guidelines (Updated)](#model-development-guidelines-updated)
* [Data Handling Guidelines](#data-handling-guidelines)
* [Testing](#testing)
* [Documentation](#documentation)
* [Submitting Changes](#submitting-changes)
* [Review Process](#review-process)
* [Model Performance Standards (Updated)](#model-performance-standards-updated)
* [Questions and Support](#questions-and-support)
* [Recognition](#recognition)
* [License](#license)

---

## 🔎 Model Development Guidelines (Updated)

### 📊 Bayesian Network Design

* **Framework**: Our model is a Directed Acyclic Graph (DAG) with nodes representing symptoms, demographics, and the target variable (`corona_result`).
* **Target Variable**: COVID-19 test result (`positive`, `negative`, `other`).
* **Structure**: Designed based on prior medical knowledge and empirical data. Any modification must maintain interpretability and alignment with clinical reasoning.

### ⚖️ Parameter Estimation

We use **Dirichlet priors** and the **BDeu scoring metric**:

* **Dirichlet Priors**: Used for smoothing and avoiding zero probabilities in CPDs.
* **BDeu Score**: Balances data likelihood and model complexity. Contributors must adhere to the derivation standards in Equations (4–10) from the published paper.

**Formula Reference:**

```math
\hat{P}(X_i = k \mid Parents(X_i) = j) = \frac{N_{ijk} + \alpha / q_i}{N_{ij} + \alpha \cdot r_i / q_i}
```

### 🤖 Inference Algorithms

Support and contributions are encouraged for:

* **Likelihood Weighting** — Preferred for real-time diagnostics due to low divergence and high accuracy.
* **Rejection Sampling** — Acceptable alternative.
* **Gibbs Sampling** — Use with caution; contributes significantly to divergence and lower efficiency.

Inference must support:

* Variable Elimination
* Marginalization
* Posterior probability estimation for queries like:

  ```math
  P(corona\_result \mid symptoms)
  ```

### 🌀 Feature Analysis & Visualization

Contributors may extend or improve visualizations like:

* **Symptom importance calculation**:

  ```math
  I(X_i) = P(Positive \mid X_i = 1) - P(Positive \mid X_i = 0)
  ```
* ROC curves per class
* Confusion matrices

> Tools like matplotlib, seaborn, or Plotly are preferred.

### 📊 Evaluation Metrics

All contributions must report:

* Accuracy, Precision, Recall, F1 Score
* ROC/AUC per class
* Divergence measures:

  * Total Variation Distance (TV Distance)
  * Hellinger Distance
  * Jensen-Shannon Divergence (JSD)
  * KL Divergence

---

## 🔄 Model Performance Standards (Updated)

To be accepted, new or modified models must maintain or exceed:

| **Metric**       | **Minimum Threshold** |
| ---------------- | --------------------- |
| Accuracy         | ≥ 94%                 |
| Precision        | ≥ 94%                 |
| Recall           | ≥ 94%                 |
| F1 Score         | ≥ 94%                 |
| AUC              | ≥ 0.94                |
| TV Distance (LW) | ≤ 0.14                |
| JSD (LW)         | ≤ 0.01                |

> ℹ️ Tip: Use **sample sizes ≥ 10,000** for robust testing of approximate inference algorithms.

---

(For the rest of the content like setup, documentation, testing, etc., see the original `CONTRIBUTING.md` file.)


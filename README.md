# λ(f): A Linearity Score for Neural Network Interpretability

This repository contains the code and datasets for the paper:

**Fidelity Isn’t Accuracy: When Linearly Decodable Functions Fail to Match the Ground Truth**  
*Jackson Eshbaugh, Lafayette College, June 2025*

---

## Overview

Neural network outputs can often be well-approximated by linear models—but what does that tell us?

This project introduces the **linearity score** λ(f), a simple metric that quantifies how well a regression network’s predictions can be mimicked by a linear model. We show that this output-level diagnostic reveals important interpretability characteristics of learned functions—especially when **fidelity** (mimicking a network) diverges from **accuracy** (matching the ground truth).

---

## Paper

📄 Read the paper: [`paper.pdf`](./paper.pdf)  
_This is a pre-submission draft. Feedback welcomed._  
Email: jacksoneshbaugh@gmail.com  
More at: [jacksoneshbaugh.github.io](https://jacksoneshbaugh.github.io)

---

## What is λ(f)?

Let \( f \) be a trained regression network, and let \( \mathcal{L} \) be the space of affine functions. Define:

\[
\lambda(f) := R^2(f, g^*) = 1 - \frac{\mathbb{E}[(f(x) - g^*(x))^2]}{\text{Var}(f(x))}
\]

where \( g^* = \arg\min_{g \in \mathcal{L}} \mathbb{E}[(f(x) - g(x))^2] \).

In other words, λ(f) measures how well a linear model can mimic the predictions of a trained neural network. Unlike typical \( R^2 \), this score is **not** about matching the ground truth—it’s about measuring how *linearly decodable* the function learned by the network is from the input space.

---

## Reproducing Results

All experiments and visualizations in the paper are contained in:

📓 [`lambda_linearity_score.ipynb`](./lambda_linearity_score.ipynb)

The notebook is fully self-contained and organized into:
- A reusable experimental framework
- Four datasets (synthetic + 3 real-world)
- Plots and tabulated results

To use λ(f) on your own data, modify the preprocessing and `build_network()` function and rerun the provided pipeline.

---

## Datasets Used

- [Medical Cost Personal Dataset (Kaggle)](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- [Concrete Compressive Strength](https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength)
- [California Housing (scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- Synthetic: \( y = x \cdot \sin(x) + \varepsilon \), where \( \varepsilon \sim \mathcal{N}(0, \sigma^2) \)

---

## Requirements

To run the notebook, install the following Python packages:

```bash
pip install tensorflow scikit-learn matplotlib pandas seaborn kagglehub
```

Tested with Python 3.11.

## Acknowledgements

Special thanks to Professor Jorge Silveyra for the early discussions that helped spark this project.

&copy; 2025 Jackson Eshbaugh. Released under the [MIT License](./LICENSE).

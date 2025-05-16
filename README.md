# λ(f): Measuring Linearity in Neural Networks

This repository contains the code and datasets for the paper:

**Fidelity Isn’t Accuracy: When Linearly Decodable Functions Fail to Match the Ground Truth**  
*Jackson Eshbaugh, Lafayette College, May 2025*

---

## Overview

Neural networks can be surprisingly well-approximated by linear models—but how do we measure that, and what does it mean?

This project introduces a simple, output-level diagnostic for regression networks: the **linearity score** $\lambda(f)$, which quantifies how well a trained neural network can be mimicked by a linear model. Our findings highlight a key tension between **fidelity** and **predictive accuracy**, especially when interpreting neural networks as black boxes.

---

## Paper

You can read the full paper here: [`paper.pdf`](./paper.pdf).
_This is a pre-submission draft. Feedback welcomed._

---

## Reproducing Results

The experiments in the paper span both synthetic and real-world datasets.

### Datasets Used:
- [Medical Cost Personal Dataset (Kaggle)](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- [Concrete Compressive Strength](https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength)
- [California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- Synthetic dataset: $y = x \cdot \sin(x) + \varepsilon$

---

## What is λ(f)?

We define the linearity score λ(f) as the coefficient of determination ( R^2 ) between a trained neural network’s output ( f(x) ) and the best-fit linear approximation ( g(x) ). Formally:

$\lambda(f) = R^2(f, g^*) = 1 - \frac{\mathbb{E}[(f(x) - g(x))^2]}{\text{Var}(f(x))}$

This score reflects how linearly decodable the learned function is from input space—without requiring access to internal activations.

---

## Acknowledgments

Special thanks to Professor Jorge Silveyra for the early discussions that helped spark this project.

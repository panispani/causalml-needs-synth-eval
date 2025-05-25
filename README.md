# Causal Machine Learning Requires Rigorous Synthetic Experiments for Broader Adoption

This repository contains the code to reproduce experiments from our paper "Causal Machine Learning Requires Rigorous Synthetic Experiments for Broader Adoption" accepted at ICML 2025.

## Overview

This repository explores some key issues in causal machine learning evaluation:

1. Examining bias in semi-synthetic datasets (Problem 2 in the paper)
2. Testing the limits beyond the identification domain (Problem 3 in the paper)

## Experiments

The repository contains two main experimental setups:

### 1. RealCause

Following the logic of Curth et al. (2021), we demonstrate that semi-synthetic methods like RealCause (Neal, 2020), which fit causal models to real data, can introduce systematic biases. Our experiments highlight how these biases can impact evaluation results.

### 2. CausalNF (Causal Normalizing Flows)

Using CausalNF (Javaloy et al., 2023), we show how studying methods beyond their identification domain can yield valuable insights. This experiment underscores the importance of conducting thorough evaluations beyond the standard identification domain of causal ML methods.

## Repository Structure

- `causal-flows/`: Code and experiments for CausalNF
- `realcause/`: Code and experiments for RealCause

Detailed setup instructions for each experiment are available in:

- `causal-flows/setup.md`: Setup and execution instructions for CausalNF experiments
- `realcause/setup.md`: Setup and execution instructions for RealCause experiments

## Citation

If you use this code in your research, please cite our paper:

```
TODO
```

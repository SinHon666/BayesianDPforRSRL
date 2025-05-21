# Bayesian Dynamic Programming for Risk-Sensitive RL

> A Bayesian DP for Risk-Sensitive Reinforcement Learning (RSRL) with support for double-layered risk measures (CVaR, Mean-Semideviation, and Expectation).

## Key Features
- **Modular Agent-Environment Interface**  
  - `Agent` and `Env` classes enable customizable transition probabilities and cost/reward matrices.
- **Double-layered Risk-Aware Bellman Operator**  
  - `RiskMeasureEst.py` module supports:
    - **Inner Risk**: Expectation, CVaR, Mean-Semideviation  
    - **Outer Risk**: Expectation, CVaR  
- **Model-Free Online Learning**  
  - Training scripts: `ChainTrain.ipynb`, `CoinTrain.ipynb`  
  - Saved policies: `./res/` directory  
- **Comprehensive Evaluation**  
  - `Evaluate.py` provides **convergence analysis** and **robustness testing**  
  - Demo evaluations: `ChainTest.ipynb`, `CoinTest.ipynb` (with visualizations)  


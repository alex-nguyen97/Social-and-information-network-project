# Social and Information Network Analysis - Assignment 3

This project explores temporal link prediction on the Astro-Ph collaboration network.
It includes:

- Graph construction and cleanup from train/test edge lists
- Structural feature engineering for candidate links
- Supervised Random Walk (SRW) implementation
- Baseline supervised machine learning models for comparison
- Visual comparison of SRW and ML recommendation scores

## Repository Structure

```
social and information network analysis/
── code/
	 ├── A3_25608100.ipynb      # Main assignment notebook
	 ├── 42913_AS3.ipynb        # Ignore for this assignment flow
	 ├── train_data.csv         # Generated during feature extraction
	 └── README.md
```

## Objectives

1. Build and analyze train/test collaboration graphs.
2. Identify future links (edges in test not present in train).
3. Train a Supervised Random Walk model for personalized link ranking.
4. Train standard ML baselines (Logistic Regression, Random Forest, Gradient Boosting).
5. Compare SRW vs ML rankings and scores.

## Environment Setup

Recommended: Python 3.10+

Install required packages:

```bash
pip install numpy pandas networkx scipy scikit-learn matplotlib
```

If you run in VS Code Notebook, select a kernel that has these packages.

## How to Run

Open:

- `code/A3_25608100.ipynb`

Run cells from top to bottom in this order:

1. Task A: Data loading, cleanup, future-edge discovery
2. Task B: Network statistics and visual summaries
3. Task C: Supervised Random Walk training/evaluation
4. Task D: ML baseline training/evaluation
5. SRW vs ML comparison table/chart

## Main Methods

### 1) Supervised Random Walk (SRW)

- Learns edge-strength weights using a logistic edge-strength function.
- Builds transition probabilities with restart probability `alpha`.
- Optimizes a WMW-style ranking loss with L-BFGS.
- Outputs personalized node ranking scores for a selected source node.

### 2) ML Baselines

Models used:

- Logistic Regression (with scaling)
- Random Forest
- Gradient Boosting

Features used per candidate pair:

- Common Neighbors
- Jaccard
- Adamic-Adar
- Resource Allocation
- Preferential Attachment

## Expected Outputs

During execution, the notebook prints:

- Node/edge counts for train and test graphs
- Number of future links
- SRW ranking quality (AUC) and top recommendations
- ML model comparison (AUC/AP) and top recommendations
- A comparison table and bar chart for SRW vs ML top recommendations

Generated file:

- `train_data.csv` in `code/`

## Troubleshooting

1. `ValueError` about tuple lengths when building dict:
   - Ensure `ranking_ml` tuples are unpacked as `(node, score, label)`.

2. No valid source node found for local positives:
   - Relax thresholds in source-selection helper (`min_global_pos`, `min_local_pos`) or increase local subgraph size.

3. Very different SRW vs ML score magnitudes:
   - This is expected. For plotting fairness, use normalization or rank-based comparison.

## Reproducibility Notes

- Random sampling uses fixed seeds where implemented (`np.random.default_rng(42)`, `random_state=42`).
- Results may still vary slightly across package versions and environments.

## Author

Assignment notebook: `A3_25608100.ipynb`

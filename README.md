# Drug Side-Effect Prediction using Machine Learning & Graph Neural Networks

Predicting the side effects of drug-drug interactions using the real-world **Decagon
polypharmacy dataset**, comparing classical machine learning approaches against a Graph
Neural Network (GNN) that leverages the actual drug-interaction network.

## Project Overview

When patients take two drugs together, the combination can cause side effects that neither
drug causes alone. This project builds a full pipeline — from raw interaction data to a
working dashboard — that predicts which side effects a given drug pair is likely to cause,
and investigates whether modeling drugs as a **network** (rather than in isolation) improves
prediction quality.

**Industry relevance:** this maps directly to pharmacovigilance / drug-safety analytics —
active work areas at organizations like ICON plc, MSD, Pfizer, and national regulators such
as the HPRA (Ireland) and EMA (EU).

## Dataset

**Decagon polypharmacy side-effect dataset** (Zitnik et al.) — a real, published drug-drug
interaction dataset.

| Property | Value |
|---|---|
| Total records | 4,649,441 |
| Unique drugs | 645 |
| Unique drug pairs (real interactions) | 63,473 |
| Unique side effects | 1,317 |
| Format | `drug_1, drug_2, side_effect_code, side_effect_name` |

For computational tractability, this project uses a **subsample** of ~97–98 drugs (selected
by moderate interaction-degree, not simply the most-connected drugs — see Methodology) and
the top 50 most common side effects.

## Methodology / Pipeline

1. **Data loading & subsampling** — real interaction pairs loaded and subsampled to a
   manageable, non-degenerate size
2. **Chemical structure retrieval** — each drug's PubChem CID is used to fetch its SMILES
   string, converted into a 512-bit Morgan fingerprint via RDKit
3. **SQL** — interaction data loaded into a SQLite database and queried directly
4. **Exploratory Data Analysis** — side-effect frequency and drug-connectivity distributions
5. **Clustering (unsupervised ML)** — K-Means groups drugs by aggregated side-effect profile
6. **Regression** — predicts a "side-effect burden" score (number of distinct side effects a
   drug is associated with) from chemical structure alone
7. **Classification** — predicts which side effects a drug **pair** causes, using Random
   Forest and a TensorFlow neural network
8. **Graph Neural Network (GAT)** — the core comparison: a Graph Attention Network trained on
   the **real drug-interaction graph** (nodes = drugs with chemical fingerprints as features,
   edges = actual recorded interactions), benchmarked against the flat-feature classifier on
   identical inputs and train/test split
9. **Consolidated metrics & reporting** — results exported to Excel and visualized in a
   Power BI dashboard

## Key Results

| Model | Task | Metric |
|---|---|---|
| Random Forest (pair-level) | Classification | AUROC 0.618, Macro F1 0.079 |
| TensorFlow MLP (pair-level) | Classification | AUROC 0.614, Macro F1 0.075 |
| Linear Regression (PCA) | Regression (side-effect burden) | R² -0.295 |
| Ridge (PCA) | Regression | R² -0.301 |
| Random Forest Regressor (PCA) | Regression | R² -0.639 |
| **Random Forest (edge/flat baseline)** | Edge Classification | **AUROC 0.628** |
| **GNN (edge/graph-aware)** | Edge Classification | **AUROC 0.674** |

**Headline finding:** the GNN outperforms the flat-feature Random Forest baseline by **4.6
AUROC points (0.674 vs. 0.628)** using *identical* chemical-fingerprint inputs and the
*identical* train/test split — demonstrating that incorporating the real drug-interaction
network provides genuine predictive value beyond chemical structure alone.

The regression task (predicting side-effect burden) produced negative R² across all three
models — reported honestly as a limitation: with a relatively small drug sample, chemical
structure alone does not strongly predict overall side-effect burden, suggesting other
factors (mechanism of action, dosage, patient population) likely matter more.

## Methodological Notes (Debugging Journey)

This project went through several rounds of diagnosis and correction, documented here for
transparency:

- **Degenerate regression target**: an early version of the "burden" target was computed only
  from the top-50 most common side effects, causing every sampled drug to touch all 50 (zero
  variance) and producing a meaningless R²=1.0. Fixed by computing burden from the full
  side-effect set.
- **Graph oversmoothing**: sampling the most-connected drugs produced a near-complete
  interaction graph (>99% density), causing the GNN's node embeddings to collapse to
  near-identical vectors. Fixed by resampling drugs with moderate (not maximal) connectivity,
  producing a properly sparse graph (~3% density).
- **Edge directionality**: the interaction graph was initially built with one-directional
  edges, silently preventing roughly half the nodes from receiving neighbor information
  during GNN training. Fixed by adding edges in both directions.
- **Class imbalance**: a per-label positive-class weight was added to the GNN's loss function
  to prevent the model from ignoring rarer side effects.

## Tech Stack

- **Languages:** Python, SQL
- **Data:** Pandas, NumPy
- **ML:** Scikit-learn (KMeans, Random Forest, Linear/Ridge Regression), TensorFlow
- **Deep Learning / GNN:** PyTorch, PyTorch Geometric (GATConv)
- **Cheminformatics:** RDKit, PubChemPy
- **Visualization:** Matplotlib, Seaborn, Power BI
- **Environment:** Google Colab / Jupyter Notebook
- **Version control:** Git / GitHub

## Repository Contents

```
drug-side-effect-prediction/
├── drug_side_effect_prediction_annotated.ipynb   # Full annotated notebook (all steps)
├── drug_report.xlsx                              # Exported results (clusters, metrics)
├── drug_dashboard.pbix                           # Power BI dashboard
├── README.md
└── screenshots/                                  # Dashboard screenshots
```

## Dashboard

The Power BI dashboard visualizes: total drugs analyzed, side effects tracked, the GNN's
headline AUROC, the most common side effects, drug cluster distribution, and a full model
comparison table with conditional formatting highlighting the GNN's best-in-class result.


## Future Work

- Scale to the full 645-drug, 63K-edge Decagon graph with more compute
- Incorporate additional drug features (target proteins, drug class) beyond chemical
  fingerprints alone
- Explore alternative GNN architectures (GraphSAGE, R-GCN for multi-relational edges)
- Address the weak regression signal with richer features or a reformulated target

## Author
Sowmiya Ramaraj

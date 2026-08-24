# MBC-STGNN

**Mass-Balance-Constrained Spatiotemporal Graph Neural Network for Riverine Total Phosphorus Reconstruction and Source Apportionment**

MBC-STGNN is a physics-guided spatiotemporal graph neural network developed for reconstructing missing total phosphorus observations in river monitoring networks and quantifying phosphorus inputs from multiple sources.

The framework integrates river-network topology, temporal dynamics, self-attention, and mass-balance constraints within a unified PyTorch model.

## Overview

Conventional data-driven models often treat monitoring stations independently or lack explicit physical constraints. MBC-STGNN addresses these limitations by combining

* **Graph Convolutional Networks (GCN)** for upstream-downstream spatial dependency modeling
* **Gated Recurrent Units (GRU)** for temporal feature extraction
* **Self-Attention** for joint spatiotemporal feature interaction
* **Mass-balance constraints** for physically consistent phosphorus reconstruction
* **Learnable physical parameters** for river transport, source delivery, and sediment-water exchange

The model is designed primarily for total phosphorus reconstruction under incomplete monitoring observations.

## Model Architecture

```text
Multivariate environmental observations
                │
                ▼
        ┌───────────────┐
        │      GRU      │
        │ Temporal      │
        │ Modeling      │
        └───────┬───────┘
                │
                │
        ┌───────▼───────┐
        │      GCN      │
        │ River-network │
        │ Modeling      │
        └───────┬───────┘
                │
                ▼
       Feature Concatenation
                │
                ▼
        Self-Attention
                │
                ▼
        Prediction Layer
                │
                ▼
    Total Phosphorus Reconstruction
                │
                ▼
      Mass-Balance Constraint
```

The river system is represented as a directed graph in which monitoring stations are graph nodes and hydrological connections define graph edges.

The training objective combines data reconstruction loss with physical regularization

```text
Total Loss
= Data Loss
+ λpde × Mass-Balance Loss
+ λnonneg × Non-Negativity Loss
+ Physical-Parameter Regularization
```

## Key Features

* Directed river-network graph representation
* 30-day temporal input window
* GCN-based spatial dependency modeling
* Two-layer GRU temporal encoder
* Joint spatiotemporal self-attention
* Mask-based missing-value reconstruction
* Mass-balance-constrained optimization
* Learnable river transport decay parameters
* Learnable phosphorus delivery coefficients
* Learnable sediment-water exchange terms
* Daily, monthly, and seasonal phosphorus source analysis

## Repository Structure

```text
MBC-STGNN/
│
├── main.py
│   └── Model training and physics-constrained optimization
│
├── stgcn.py
│   └── GCN, GRU, Self-Attention, and MBC-STGNN architecture
│
├── utils.py
│   └── Data loading, normalization, graph processing, and mask generation
│
├── predict_timeseries.py
│   └── Model evaluation and time-series reconstruction
│
├── predict_full_series_imputation.py
│   └── Full-series TP imputation
│
├── plot_gt_vs_imputed_full_pipeline.py
│   └── Visualization of observations and reconstructed values
│
├── 1 calculate_daily_input.py
│   └── Daily phosphorus source calculation
│
├── 2 calculate_monthly_input.py
│   └── Monthly phosphorus source aggregation
│
├── 3 calculate_average_phosphorus.py
│   └── Station-level mean phosphorus source calculation
│
├── 4 calculate_phosphorus_seasonal_average.py
│   └── Dry- and wet-season phosphorus source analysis
│
├── 5 phosphorus_ratio_clustering.py
│   └── Hierarchical clustering based on phosphorus source composition
│
├── requirements.txt
└── .gitignore
```

## Environment

The project was developed with Python and PyTorch.

Main dependencies include

```text
PyTorch
NumPy
Pandas
Matplotlib
SciPy
scikit-learn
```

Install the required packages with

```bash
pip install -r requirements.txt
```

GPU acceleration is automatically enabled when CUDA is available.

## Data

The datasets used in this project are not included in the repository because of their size and data-management requirements.

The model expects a `data/` directory containing the processed monitoring, environmental, hydrological, river-network, and phosphorus-source data.

Representative inputs include

```text
data/
├── adj_mat.npy
├── node_values.npy
├── A_upstream.npy
├── D_upstream.npy
├── Q_node_daily.npy
├── L_industry_daily.npy
├── L_crop_farming_daily.npy
├── L_livestock_breeding_daily.npy
├── L_urban_life_daily.npy
└── L_rural_life_daily.npy
```

`node_values.npy` contains node-level multivariate environmental features. The current implementation uses total phosphorus (`Water_TP`) as the reconstruction target.

The input features include water-quality variables, meteorological variables, hydrological variables, soil properties, land-use variables, agricultural inputs, and socioeconomic indicators.

## Training

Run

```bash
python main.py
```

The default configuration includes

```text
Input window        30 days
Output window       30 days
Mask ratio          0.30
Epochs              100
Batch size          32
Random seed         7
Physics loss weight 0.01
```

During training, part of the observed TP values is randomly masked. The model reconstructs these artificially missing observations while simultaneously minimizing the mass-balance residual.

## Inference

Time-series reconstruction and evaluation can be performed with

```bash
python predict_timeseries.py
```

Full-series imputation can be performed with

```bash
python predict_full_series_imputation.py
```

The reconstructed series can be compared with observations using

```bash
python plot_gt_vs_imputed_full_pipeline.py
```

## Phosphorus Source Analysis

After reconstruction, the source-apportionment workflow can be executed sequentially

```bash
python "1 calculate_daily_input.py"
python "2 calculate_monthly_input.py"
python "3 calculate_average_phosphorus.py"
python "4 calculate_phosphorus_seasonal_average.py"
python "5 phosphorus_ratio_clustering.py"
```

The analysis considers phosphorus inputs from

* Upstream river transport
* Industrial wastewater
* Urban domestic wastewater
* Crop farming
* Livestock breeding
* Rural domestic wastewater
* Sediment-water exchange

Monthly and seasonal analyses can subsequently be used to examine spatial and seasonal shifts in phosphorus-source dominance.

## Performance

In the source-basin experiments, MBC-STGNN achieved

| Metric | Performance |
| ------ | ----------: |
| NSE    |        0.89 |
| RMSE   |   0.02 mg/L |

The model outperformed conventional RNN, GRU, and GCN baseline models while maintaining explicit physical consistency through the mass-balance constraint.

## Notes

This repository currently focuses on the MBC-STGNN reconstruction and phosphorus source-apportionment pipeline.

Large datasets, intermediate results, and trained model checkpoints are excluded from version control.

Some post-processing scripts may require adjustment of local data paths before execution on a new machine.

## License

A license has not yet been specified for this repository.

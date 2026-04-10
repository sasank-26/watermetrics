# 🌊 Hybrid Spatiotemporal Transformer-GNN for Water Quality Prediction

> **A Research-Level Framework for Predictive Water Quality Modeling**

This project provides an end-to-end, publication-ready machine learning pipeline for spatiotemporal water quality forecasting. It integrates the power of **Graph Neural Networks (GNN)** for spatial geographic dependencies and **Transformer Encoders** for complex temporal patterns, complete with comprehensive baselines, a real-world Water Quality Index (WQI) calculator, hyperparameter tuning, and a beautiful interactive Streamlit dashboard.

## 🚀 Key Features

*   🧠 **Hybrid Architecture**: Combines KNN-based Graph Layers (for learning pollution propagation across geographic stations) with Multi-Head Self-Attention Transformers (for capturing long-term temporal and seasonal dependencies).
*   📊 **Comprehensive Model Comparison**: Automatically trains and evaluates the Hybrid Model against strong baselines (**Random Forest**, **LSTM**, and **Standalone GNN**) using MAE, RMSE, R², and MAPE metrics.
*   💧 **Water Quality Index (WQI)**: Integrates China's GB 3838-2002 standards via a weighted-arithmetic sub-index formulas to convert multi-parameter chemical predictions into an actionable index out of 100 with an Early Warning System.
*   🔬 **Publication-Quality Visualizations**: Automatically generates clean, research-ready PNG plots including Actual vs Predicted scatter plots, Time trends, Error distributions, Feature importance (RF), and Loss curves.
*   🖥️ **Interactive Streamlit Dashboard**: A full GUI to explore performance metrics, time-series forecasting trends, WQI calculators, and architectural breakdowns.
*   🎛️ **Optuna Hyperparameter Tuning**: Included optimization script for fine-tuning the model's structural parameters (number of heads, layers, dropout) and optimization variables (learning rate, weight decay).

---

## 🏗️ Model Architecture

The core of this project is the **HybridSTModel**, addressing the deficiencies of standard sequential models (like LSTM) by treating monitoring stations as a geographic graph.

1.  **Input Data**: Merges time-series data from `monthly_ocean.csv` and `weekly_land.csv`.
2.  **Temporal Learning (Transformer)**: Passes each station's sequence window through a Multi-Head Attention layer to capture complex time-based dependencies.
3.  **Spatial Learning (GNN)**: A K-Nearest Neighbors (KNN) adjacency matrix with RBF kernel weighting builds real physical connectivity between stations. The final temporal states propagate over this graph, capturing the spread of water quality conditions.
4.  **Fusion Layer**: Concatenates spatial and temporal extracted features, passing them through dense projection layers with GELU activation to output the target indicators (e.g., DO, pH).

---

## 🛠️ Installation

**Prerequisites:** Python 3.10+

1. Clone the repository and navigate into the folder:
   ```bash
   git clone https://github.com/sasank-26/watermetrics.git
   cd watermetrics
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 Usage Guide

### 1. Run the Full Model Comparison Pipeline
This script trains all sub-models (Random Forest, LSTM, GNN, Hybrid), formats a comparison table, and generates the publication plots inside the `outputs/` folder.

```bash
# Recommended parameters for full training
python run_comparison.py --epochs 30 --batch-size 32
```
*Check the `outputs/` folder for `.png` graphs and `comparison_table.csv`.*

### 2. Launch the Interactive Dashboard
To visually explore the results, investigate the forecast trends, or calculate WQI with the Early Warning System:

```bash
streamlit run app.py
```

### 3. Hyperparameter Tuning
Optimize the Hybrid model's architecture using Optuna:

```bash
python src/wq_hybrid/tuning.py --trials 20
```
*The best hyperparameter configuration will be saved to `outputs/best_hyperparameters.json`.*

### 4. Basic Single Model Training (Legacy)
If you only want to quickly train the Hybrid model:

```bash
python run_training.py --epochs 20
```

---

## 📂 Project Structure

```text
.
├── src/wq_hybrid/
│   ├── __init__.py       # Module exports
│   ├── baselines.py      # LSTM, RF, and Standalone GNN architectures
│   ├── compare.py        # Pipeline running all models & calculating metrics
│   ├── config.py         # Centralized configuration DataClass
│   ├── data.py           # Preprocessing, data merging, & tensor construction
│   ├── graph.py          # KNN layout calculation and Adjacency graphs
│   ├── model.py          # Core Hybrid GNN-Transformer Architecture
│   ├── train.py          # Legacy training loops
│   ├── tuning.py         # Optuna hyperparameter optimization script
│   ├── visualize.py      # Code to generate Matplotlib research visuals
│   └── wqi.py            # Real-world Water Quality Index formulas
├── app.py                # Streamlit Dashboard UI
├── run_comparison.py     # Main CLI entry point to run paper experiments
├── run_training.py       # Simple CLI for one-off training
├── requirements.txt      # Dependency list
├── README.md             # This file
├── monthly_ocean.csv     # (Dataset) Ocean metric data
└── weekly_land.csv       # (Dataset) Land metric data
```

---

## 🧪 Scientific Implementation Details

### Indicators Used
The system natively tracks 8 important metrics: `DO, pH, COD, CODMn, NH4N, DIN, DIP, TPH` and by default aims to forecast `DO` and `pH`. You can modify the predicted attributes using the command flags.

### WQI Limits (GB 3838-2002)
Our WQI logic checks thresholds for standard drinking and environment water: Minimum DO (5.0 mg/L), maximum COD (20.0 mg/L), maximum pH limits (6.0 - 9.0), etc.

### Evaluation Metrics
Computed automatically per model:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)
- **R²** (Coefficient of Determination)
- **MAPE** (Mean Absolute Percentage Error)
- **Training Time**

---

*This codebase provides a highly extensible foundation; further topological directed edges (like river-flow data) or climate sequence variables can easily be integrated into the data-loader for additional research scope.*

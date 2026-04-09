"""Streamlit Dashboard for Water Quality Prediction.

Run with:
    streamlit run app.py

Features:
  - Model comparison table & charts
  - Actual vs Predicted visualizations
  - Loss curves
  - Feature importance
  - WQI calculator & time series
  - Station-wise prediction explorer
  - Multi-step forecasting view
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Add project root to path ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wq_hybrid.wqi import compute_wqi, wqi_category, STANDARDS, DEFAULT_WEIGHTS

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Water Quality Prediction Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        color: white;
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(0,0,0,0.05);
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-card h3 {
        font-size: 0.85rem;
        color: #7f8c8d;
        margin: 0 0 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c3e50;
    }
    
    .wqi-excellent { color: #27ae60; font-weight: 700; }
    .wqi-good { color: #2ecc71; font-weight: 700; }
    .wqi-medium { color: #f39c12; font-weight: 700; }
    .wqi-bad { color: #e67e22; font-weight: 700; }
    .wqi-very-bad { color: #e74c3c; font-weight: 700; }
    
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #2c3e50;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ── Cache helpers ────────────────────────────────────────────────────────────
OUTPUT_DIR = ROOT / "outputs"

@st.cache_data
def load_comparison_results():
    path = OUTPUT_DIR / "comparison_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_data
def load_comparison_table():
    path = OUTPUT_DIR / "comparison_table.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return None

@st.cache_data
def load_predictions(model_key: str):
    true_path = OUTPUT_DIR / f"pred_{model_key}_true.npy"
    pred_path = OUTPUT_DIR / f"pred_{model_key}_pred.npy"
    if true_path.exists() and pred_path.exists():
        return np.load(true_path), np.load(pred_path)
    return None, None

@st.cache_data
def load_feature_importances():
    path = OUTPUT_DIR / "rf_feature_importances.npy"
    if path.exists():
        return np.load(path)
    return None

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🌊 Water Quality Prediction Dashboard</h1>
    <p>Hybrid Spatiotemporal Transformer-GNN Model for Water Quality Prediction</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/water.png", width=60)
    st.title("Navigation")
    page = st.radio(
        "Go to",
        [
            "📊 Model Comparison",
            "📈 Predictions & Trends",
            "🔬 Feature Analysis",
            "💧 WQI Calculator",
            "🏗️ Architecture",
        ],
    )
    st.divider()
    st.caption("Hybrid Spatiotemporal Transformer-GNN")
    st.caption("© 2025 Water Quality Research")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Model Comparison
# ═════════════════════════════════════════════════════════════════════════════
if page == "📊 Model Comparison":
    st.markdown('<div class="section-header">📊 Model Performance Comparison</div>', unsafe_allow_html=True)

    results = load_comparison_results()
    table = load_comparison_table()

    if results and "comparison" in results:
        comp = results["comparison"]
        models = list(comp.keys())

        # Metric cards for the best model (Hybrid)
        if "Hybrid Transformer-GNN" in comp:
            hybrid = comp["Hybrid Transformer-GNN"]
            cols = st.columns(4)
            for col, (metric, val) in zip(cols, [
                ("MAE", hybrid.get("MAE", "N/A")),
                ("RMSE", hybrid.get("RMSE", "N/A")),
                ("R² Score", hybrid.get("R²", "N/A")),
                ("MAPE (%)", hybrid.get("MAPE(%)", "N/A")),
            ]):
                with col:
                    v = f"{val:.4f}" if isinstance(val, float) else str(val)
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{metric}</h3>
                        <div class="value">{v}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("*<center>Hybrid Transformer-GNN (Proposed Model) Metrics</center>*",
                       unsafe_allow_html=True)

        st.markdown("")

        # Comparison table
        if table is not None:
            st.subheader("Full Comparison Table")
            st.dataframe(
                table.style.highlight_min(subset=["MAE", "RMSE"], color="#d4efdf")
                          .highlight_max(subset=["R²"], color="#d4efdf")
                          .format("{:.4f}"),
                use_container_width=True,
            )

        # Bar charts
        st.subheader("Visual Comparison")
        tab1, tab2, tab3 = st.tabs(["MAE", "RMSE", "R² Score"])

        import plotly.graph_objects as go

        color_map = {
            "Random Forest": "#e74c3c",
            "LSTM": "#3498db",
            "Standalone GNN": "#2ecc71",
            "Hybrid Transformer-GNN": "#9b59b6",
        }

        for tab, metric in zip([tab1, tab2, tab3], ["MAE", "RMSE", "R²"]):
            with tab:
                vals = [comp[m].get(metric, 0) for m in models]
                colors = [color_map.get(m, "#95a5a6") for m in models]

                fig = go.Figure(data=[go.Bar(
                    x=models, y=vals,
                    marker_color=colors,
                    text=[f"{v:.4f}" for v in vals],
                    textposition="outside",
                )])
                fig.update_layout(
                    title=f"{metric} Comparison",
                    yaxis_title=metric,
                    template="plotly_white",
                    height=400,
                    font=dict(family="Inter"),
                )
                st.plotly_chart(fig, use_container_width=True)

        # Loss curves
        loss_curves = results.get("loss_curves", {})
        if loss_curves:
            st.subheader("Training & Validation Loss Curves")
            cols = st.columns(len(loss_curves))
            for col, (model_name, curves) in zip(cols, loss_curves.items()):
                with col:
                    fig = go.Figure()
                    epochs = list(range(1, len(curves["train"]) + 1))
                    color = color_map.get(model_name, "#3498db")
                    fig.add_trace(go.Scatter(
                        x=epochs, y=curves["train"],
                        mode="lines+markers", name="Train",
                        line=dict(color=color, width=2),
                    ))
                    fig.add_trace(go.Scatter(
                        x=epochs, y=curves["val"],
                        mode="lines+markers", name="Validation",
                        line=dict(color=color, width=2, dash="dash"),
                    ))
                    fig.update_layout(
                        title=model_name,
                        xaxis_title="Epoch",
                        yaxis_title="MSE Loss",
                        template="plotly_white",
                        height=350,
                        font=dict(family="Inter"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("⚠️ No comparison results found. Run the comparison pipeline first:\n\n"
                   "```bash\npython run_comparison.py --epochs 30\n```")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Predictions & Trends
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈 Predictions & Trends":
    st.markdown('<div class="section-header">📈 Predictions & Time Trends</div>', unsafe_allow_html=True)

    model_options = {
        "Random Forest": "random_forest",
        "LSTM": "lstm",
        "Standalone GNN": "standalone_gnn",
        "Hybrid Transformer-GNN": "hybrid_transformer-gnn",
    }

    selected_model = st.selectbox("Select Model", list(model_options.keys()), index=3)
    model_key = model_options[selected_model]
    y_true, y_pred = load_predictions(model_key)

    if y_true is not None:
        yt = y_true.reshape(-1, y_true.shape[-1]) if y_true.ndim > 2 else y_true
        yp = y_pred.reshape(-1, y_pred.shape[-1]) if y_pred.ndim > 2 else y_pred
        target_names = ["DO", "pH"]  # default targets

        tab1, tab2, tab3 = st.tabs(["📉 Time Series", "🎯 Scatter Plot", "📊 Error Distribution"])

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        color = {"Random Forest": "#e74c3c", "LSTM": "#3498db",
                 "Standalone GNN": "#2ecc71", "Hybrid Transformer-GNN": "#9b59b6"}.get(selected_model, "#3498db")

        with tab1:
            n_show = st.slider("Points to display", 50, min(500, yt.shape[0]), min(200, yt.shape[0]))
            for i, tname in enumerate(target_names):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=yt[:n_show, i], mode="lines", name="Actual",
                    line=dict(color="#2c3e50", width=2)))
                fig.add_trace(go.Scatter(
                    y=yp[:n_show, i], mode="lines", name="Predicted",
                    line=dict(color=color, width=2, dash="dash")))
                fig.update_layout(
                    title=f"{tname} — {selected_model}",
                    xaxis_title="Sample", yaxis_title=tname,
                    template="plotly_white", height=400,
                    font=dict(family="Inter"),
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            for i, tname in enumerate(target_names):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=yt[:, i], y=yp[:, i],
                    mode="markers", name="Predictions",
                    marker=dict(color=color, size=4, opacity=0.4)))
                lims = [min(yt[:, i].min(), yp[:, i].min()),
                        max(yt[:, i].max(), yp[:, i].max())]
                fig.add_trace(go.Scatter(
                    x=lims, y=lims, mode="lines", name="Perfect",
                    line=dict(color="black", dash="dash")))
                fig.update_layout(
                    title=f"Actual vs Predicted — {tname}",
                    xaxis_title="Actual", yaxis_title="Predicted",
                    template="plotly_white", height=450,
                    font=dict(family="Inter"),
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            for i, tname in enumerate(target_names):
                errors = yt[:, i] - yp[:, i]
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=errors, nbinsx=50,
                    marker_color=color, opacity=0.7))
                fig.add_vline(x=0, line_dash="dash", line_color="black")
                fig.update_layout(
                    title=f"Error Distribution — {tname}",
                    xaxis_title="Error (Actual - Predicted)",
                    yaxis_title="Count",
                    template="plotly_white", height=400,
                    font=dict(family="Inter"),
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No prediction data found for this model. Run comparison first.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Feature Analysis
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Feature Analysis":
    st.markdown('<div class="section-header">🔬 Feature Analysis</div>', unsafe_allow_html=True)

    fi = load_feature_importances()
    if fi is not None:
        feature_names = ["DO", "pH", "COD", "CODMn", "NH4N", "DIN", "DIP", "TPH"]
        n_features = len(feature_names)
        seq_len = fi.shape[0] // n_features if fi.shape[0] % n_features == 0 else 8

        # Aggregated importance
        imp_matrix = fi.reshape(seq_len, n_features)
        agg_imp = imp_matrix.sum(axis=0)
        agg_imp = agg_imp / agg_imp.sum()

        sorted_idx = np.argsort(agg_imp)[::-1]

        import plotly.graph_objects as go

        st.subheader("Feature Importance (Random Forest)")

        fig = go.Figure(data=[go.Bar(
            x=[feature_names[i] for i in sorted_idx],
            y=agg_imp[sorted_idx],
            marker_color=[f"hsl({h}, 70%, 50%)" for h in np.linspace(200, 300, n_features)],
            text=[f"{v:.3f}" for v in agg_imp[sorted_idx]],
            textposition="outside",
        )])
        fig.update_layout(
            yaxis_title="Importance",
            template="plotly_white", height=450,
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Temporal importance heatmap
        st.subheader("Temporal Feature Importance Heatmap")
        imp_norm = imp_matrix / imp_matrix.sum()

        fig = go.Figure(data=go.Heatmap(
            z=imp_norm.T,
            x=[f"t-{seq_len - i}" for i in range(seq_len)],
            y=feature_names,
            colorscale="Viridis",
            text=np.round(imp_norm.T, 4),
            texttemplate="%{text}",
        ))
        fig.update_layout(
            title="Feature Importance Across Time Steps",
            xaxis_title="Time Step",
            yaxis_title="Feature",
            template="plotly_white", height=400,
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("⚠️ No feature importance data. Run comparison first.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: WQI Calculator
# ═════════════════════════════════════════════════════════════════════════════
elif page == "💧 WQI Calculator":
    st.markdown('<div class="section-header">💧 Water Quality Index Calculator</div>', unsafe_allow_html=True)

    st.markdown("""
    The **Water Quality Index (WQI)** is a single number (0–100) that summarizes 
    the overall water quality from multiple parameters. Based on China's 
    **GB 3838-2002** surface water quality standards.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Enter Parameter Values")
        params = {}
        params["DO"] = st.number_input("Dissolved Oxygen (mg/L)", min_value=0.0, max_value=20.0, value=7.0, step=0.1)
        params["pH"] = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.2, step=0.1)
        params["COD"] = st.number_input("Chemical Oxygen Demand (mg/L)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
        params["CODMn"] = st.number_input("CODMn (mg/L)", min_value=0.0, max_value=50.0, value=3.0, step=0.1)
        params["NH4N"] = st.number_input("Ammonia Nitrogen (mg/L)", min_value=0.0, max_value=10.0, value=0.3, step=0.01)
        params["DIN"] = st.number_input("Dissolved Inorganic Nitrogen (mg/L)", min_value=0.0, max_value=5.0, value=0.2, step=0.01)
        params["DIP"] = st.number_input("Dissolved Inorganic Phosphorus (mg/L)", min_value=0.0, max_value=1.0, value=0.01, step=0.001, format="%.3f")
        params["TPH"] = st.number_input("Total Petroleum Hydrocarbons (mg/L)", min_value=0.0, max_value=1.0, value=0.02, step=0.001, format="%.3f")

    with col2:
        st.subheader("WQI Result")

        wqi_score = compute_wqi(params)
        category = wqi_category(wqi_score)

        # Color mapping
        cat_colors = {
            "Excellent": ("#27ae60", "🟢"),
            "Good": ("#2ecc71", "🟢"),
            "Medium": ("#f39c12", "🟡"),
            "Bad": ("#e67e22", "🟠"),
            "Very Bad": ("#e74c3c", "🔴"),
        }
        color, emoji = cat_colors.get(category, ("#95a5a6", "⚪"))

        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, {color}22 0%, {color}11 100%); 
                    border-radius: 16px; border: 2px solid {color};">
            <div style="font-size: 4rem; font-weight: 800; color: {color};">{wqi_score:.1f}</div>
            <div style="font-size: 1.5rem; font-weight: 600; color: {color};">{emoji} {category}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown("#### WQI Scale")
        scale_data = pd.DataFrame({
            "Range": ["90-100", "70-89", "50-69", "25-49", "0-24"],
            "Category": ["Excellent", "Good", "Medium", "Bad", "Very Bad"],
            "Description": [
                "Safe for drinking", "Minor treatment needed",
                "Moderate treatment needed", "Heavily polluted", "Severely polluted"
            ],
        })
        st.table(scale_data)

        st.markdown("#### Sub-Index Breakdown")
        from wq_hybrid.wqi import _sub_index
        sub_indices = {k: _sub_index(k, v) for k, v in params.items()}
        si_df = pd.DataFrame({
            "Parameter": list(sub_indices.keys()),
            "Value": [params[k] for k in sub_indices],
            "Sub-Index": [f"{v:.1f}" for v in sub_indices.values()],
            "Weight": [f"{DEFAULT_WEIGHTS.get(k, 0):.2f}" for k in sub_indices],
        })
        st.dataframe(si_df, use_container_width=True)

    # Early Warning System
    st.divider()
    st.subheader("🚨 Early Warning System")

    warnings = []
    if params["DO"] < 5.0:
        warnings.append(("🔴", "DO", f"Dissolved Oxygen ({params['DO']:.1f} mg/L) is below safe threshold (5.0 mg/L)"))
    if params["pH"] < 6.0 or params["pH"] > 9.0:
        warnings.append(("🔴", "pH", f"pH ({params['pH']:.1f}) is outside safe range (6.0-9.0)"))
    if params["NH4N"] > 1.0:
        warnings.append(("🟠", "NH4N", f"Ammonia Nitrogen ({params['NH4N']:.2f} mg/L) exceeds standard (1.0 mg/L)"))
    if params["COD"] > 20.0:
        warnings.append(("🟠", "COD", f"COD ({params['COD']:.1f} mg/L) exceeds standard (20.0 mg/L)"))
    if params["TPH"] > 0.05:
        warnings.append(("🔴", "TPH", f"Total Petroleum Hydrocarbons ({params['TPH']:.3f} mg/L) exceeds standard (0.05 mg/L)"))

    if warnings:
        for emoji, param, msg in warnings:
            st.warning(f"{emoji} **{param}**: {msg}")
    else:
        st.success("✅ All parameters are within safe limits!")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Architecture
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🏗️ Architecture":
    st.markdown('<div class="section-header">🏗️ Model Architecture</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Hybrid Spatiotemporal Transformer-GNN Architecture
    
    The proposed model combines **Graph Neural Networks** for spatial learning 
    with **Transformer Encoders** for temporal pattern recognition.
    """)

    # Architecture diagram using mermaid
    st.markdown("""
    ```
    ┌──────────────────────────────────────────────────────┐
    │                   INPUT DATA                          │
    │         [Batch, SeqLen, Nodes, Features]              │
    └──────────────────────┬───────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Input Proj  │  Linear(F → D)
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │   TRANSFORMER ENCODER    │
              │  (Multi-Head Attention)  │
              │  • Temporal Dependencies │
              │  • Seasonal Patterns     │
              │  • Long-range Context    │
              └────────────┬────────────┘
                           │
                    [B, N, D] (last step)
                           │
              ┌────────────▼────────────┐
              │    GNN SPATIAL LAYERS    │
              │  • KNN-based Adjacency   │
              │  • Message Passing       │
              │  • Pollution Propagation │
              └────────────┬────────────┘
                           │
            ┌──────────────▼──────────────┐
            │      FUSION LAYER            │
            │  concat(temporal, spatial)   │
            │  → Linear(2D → D) → GELU    │
            └──────────────┬──────────────┘
                           │
                    ┌──────▼──────┐
                    │  Dense Head  │  Linear(D → O)
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │    PREDICTION OUTPUT     │
              │  [Batch, Nodes, Targets] │
              │  (DO, pH, etc.)          │
              └─────────────────────────┘
    ```
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### 🔗 GNN (Spatial)
        - KNN-based adjacency matrix
        - RBF kernel edge weights
        - Symmetric normalization
        - Multi-layer message passing
        - Models station relationships
        """)
    with col2:
        st.markdown("""
        #### ⏱️ Transformer (Temporal)
        - Multi-head self-attention
        - Positional encoding
        - GELU activation
        - Captures seasonality
        - Long-range dependencies
        """)
    with col3:
        st.markdown("""
        #### 🔀 Fusion Layer
        - Concatenates spatial + temporal
        - Dense projection
        - GELU non-linearity
        - Dropout regularization
        - Final prediction head
        """)

    st.divider()

    st.markdown("""
    ### Model Configuration
    """)

    config_data = {
        "Parameter": ["d_model", "nhead", "Transformer Layers", "GNN Layers", 
                       "Dropout", "KNN K", "Sequence Length", "Optimizer"],
        "Value": ["64", "4", "2", "2", "0.1", "5", "8", "AdamW"],
        "Description": [
            "Hidden dimension size",
            "Number of attention heads",
            "Transformer encoder layers",
            "Graph neural network layers",
            "Regularization dropout rate",
            "K-nearest neighbors for graph",
            "Input time window length",
            "Optimizer with weight decay"
        ]
    }
    st.table(pd.DataFrame(config_data))

    st.markdown("""
    ### Key Innovations
    
    1. **Spatial-Temporal Fusion**: Unlike standalone models, our hybrid approach 
       captures both geographic station relationships and temporal water quality trends.
    
    2. **KNN Graph Construction**: Uses RBF kernel on geographic coordinates to build 
       a realistic station connectivity graph (not a simple chain).
    
    3. **Transformer over LSTM**: Multi-head self-attention captures long-range 
       seasonal patterns better than recurrent models.
    
    4. **Water Quality Index**: Integrated WQI calculation transforms multi-parameter 
       predictions into actionable water quality assessments.
    """)

# Physics-Informed Digital Twin for LSTM-Based Wind Power Forecasting

This repository contains the official implementation and validation framework for the paper: 
**"Physics-Informed Digital Twin for Pre-Deployment Validation of LSTM-Based Wind Power Forecasting Models"** accepted for presentation at **OSMSES 2026** (Karlsruhe, Germany).

---

## 🚀 Overview
We introduce a verification layer that decouples algorithmic performance from SCADA data-quality issues. By utilizing a physics-based Digital Twin engine, we establish a "Synthetic Ground Truth" to validate LSTM architectures before they are deployed to real-world wind parks.

### Key Results
- **nRMSE:** 2.25% (Normalized to 2.3 MW rated capacity)
- **Model Architecture:** Stacked LSTM (128 units, 64 units)
- **Input Vector:** $X_t = [v_t, T_t, P_{t-1}, H_t, M_t, \mu_{3h}]$

---

## 📂 Repository Structure
* **`/src`**: Contains the data generation engine and the unified forecasting pipeline.
* **`/data`**: Contains the `Synthetic_Bogdanci_Data.csv` dataset.
* **`README.md`**: Project documentation and methodology.

---

## 🛠️ Installation & Usage

1. **Clone the repository and install dependencies:**
   ```bash
   pip install -r requirements.txt
   
2. **Run the unified forecasting pipeline:**
This script will train the LSTM from scratch on your local machine and generate the Comparative Analysis (Figure 4) used in the paper:

 python src/lstm_forecasting_pipeline.py


## Methodology
- **Cyclic Time Encoding:** Preserving temporal periodicity for Hour ($H_t$) and Month ($M_t$) using sine and cosine transformations to ensure the model understands time-of-day and seasonal patterns.
- **Physics-Based Constraints:** Piecewise cubic power curve modeling to ensure the synthetic ground truth respects the aerodynamic laws specific to the Bogdanci Wind Park.
- **Signal Processing:** 3-hour rolling mean filters ($\mu_{3h}$) to attenuate high-frequency turbulence and improve LSTM convergence during training.

## Citation
If you use this work or the Digital Twin engine in your research, please cite:

**Ramadani, B. (2026).** Physics-Informed Digital Twin for Pre-Deployment Validation of LSTM-Based Wind Power Forecasting Models. OSMSES 2026, Karlsruhe, Germany.

© 2026 Blerant Ramadani - Mother Teresa University in Skopje







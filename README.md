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

   -----

## Run the unified forecasting pipeline:
This script will train the LSTM from scratch on your local machine and generate the Comparative Analysis (Figure 4):

```Bash

python src/lstm_forecasting_pipeline.py

## MethodologyThe pipeline implements:Cyclic Time Encoding: Preserving temporal periodicity for Hour ($H_t$) and Month ($M_t$).Physics-Based Constraints: Piecewise cubic power curve modeling.Signal Processing: 3-hour rolling mean filters ($\mu_{3h}$) to attenuate turbulence.

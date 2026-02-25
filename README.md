# Physics-Informed Digital Twin for LSTM-Based Wind Power Forecasting

This repository contains the official implementation and validation framework for the paper: 
**"Physics-Informed Digital Twin for Pre-Deployment Validation of LSTM-Based Wind Power Forecasting Models"** accepted for presentation at **OSMSES 2026** (Karlsruhe, Germany).

## 🚀 Overview
We introduce a verification layer that decouples algorithmic performance from SCADA data-quality issues. By utilizing a physics-based Digital Twin engine, we establish a "Synthetic Ground Truth" to validate LSTM architectures before they are deployed to real-world wind parks.

### Key Results
- **nRMSE:** 2.25% (Normalized to 2.3 MW rated capacity)
- **Model Architecture:** Stacked LSTM (128 units, 64 units)
- **Input Vector:** Xt = [vt, Tt, Pt-1, Ht, Mt, mu_3h]

## 📂 Repository Structure
* `/src`: Contains the data generation engine and the forecasting pipeline.
* `/data`: Contains the synthetic Bogdanci wind park dataset.
* `/models`: Contains the pre-trained weights (.h5 file) for immediate result verification.

## 🛠️ Installation & Usage
1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt

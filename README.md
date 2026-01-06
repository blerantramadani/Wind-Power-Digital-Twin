# Wind-Power-Digital-Twin
# Physics-Informed Digital Twin for Wind Power Forecasting

**Conference:** OSMSES 2026 (Karlsruhe, Germany)
**Status:** Submitted for Review

## Overview
This repository contains the source code for the "Digital Twin" validation framework. It includes a physics-based simulation engine that generates synthetic SCADA data for a 2.3 MW wind turbine, respecting aerodynamic constraints and atmospheric autocorrelation.

## Repository Structure
* `generate_data.py`: Physics Engine. Generates synthetic wind/power data using Random Walk logic.
* `final_optimization.py`: AI Engine. Trains the LSTM network (TensorFlow/Keras).
* `final_paper_graph.py`: Validation Engine. Generates the dynamic response graphs (Figure 3 in the paper).

## Usage
1. Run `generate_data.py` to create the dataset.
2. Run `final_paper_graph.py` to train the model and visualize results.

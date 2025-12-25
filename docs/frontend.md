# Momemura Frontend

The Momemura Frontend is an interactive visualization tool built with [Streamlit](https://streamlit.io/) to explore the results of MOME optimizations.

## Overview

The frontend allows you to:
- **Browse Runs**: Select from available optimization runs in the `output/` directory.
- **Analyze Pareto Fronts**: Interact with the Global Pareto Front scatter plot to identify trade-offs between Expectation Value and Probability.
- **Explore Cells**: View the "Best Expectation Value per Cell" heatmap and drill down into local Pareto fronts for specific regions of the search space (defined by Complexity and Total Photons).
- **Inspect Solutions**: Select individual points to view:
    - Detailed metrics (Expectation, LogProb, etc.)
    - Reconstructed **Circuit Diagrams** (using `GaussianHeraldCircuit`).
    - **Wigner Functions** (High-Definition Heatmaps):
        - **Simulated State**: The output state of the circuit.
        - **Target State**: The ground state of the target GKP operator (for side-by-side comparison).
        - **Visuals**: Uses a custom plateau colormap to cleanly separate positive/negative regions from vacuum noise.

## Automatic Parameter Parsing
The frontend automatically detects the **Target GKP parameters** (`alpha`, `beta`) from the **folder name** of the selected run (e.g., `run_alpha_2.0_beta_0.0`). This ensures the "Target State" visualization always matches the optimization intent, even if the config file is generic.

## Installation

Ensure you have installed the required dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `streamlit`
- `plotly`
- `pandas`
- `qutip` (for Wigner function calculation)
- `thewalrus` (for state preparation)

## Usage

1.  **Run the Streamlit App**:
    From the project root directory, run:
    ```bash
    streamlit run frontend/app.py
    ```

2.  **Select a Run**:
    Use the sidebar to choose an optimization run. By default, the most recent run is loaded.

3.  **Interact with Plots**:
    - **Global Pareto Front (Left)**: Hover over points to see details. Click a point to select it.
    - **Heatmap (Right)**: Click a cell to view the local Pareto front for that cell.

4.  **View Details**:
    - **Selected Solution**: If you click a point on the Pareto front, a "Selected Solution Details" section will appear below. It shows the circuit diagram and the Wigner function of the state.
    - **Selected Cell**: If you click a cell on the heatmap, a "Selected Cell Details" section will appear, showing all valid solutions found within that cell.

## Troubleshooting

-   **No Runs Found**: Ensure you have run an optimization script (e.g., `python run_mome.py`) and that it generated an `output/` directory with `results.pkl`.
-   **Missing Dependencies**: If you see `ModuleNotFoundError`, check that your virtual environment is active and up-to-date with `requirements.txt`.

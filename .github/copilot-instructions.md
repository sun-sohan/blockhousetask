# Copilot Instructions for AI Coding Agents

## Project Overview
This project computes Order Flow Imbalance (OFI) features from limit order book data, using pandas and scikit-learn. The main workflow is in `main.py`, which processes a CSV file and outputs engineered features to the `feature_outputs/` directory.

## Key Files & Structure
- `main.py`: Core logic for OFI feature engineering, including best level, multi-level, integrated, and cross-asset OFI calculations.
- `first_25000_rows.csv`: Input data (limit order book events).
- `feature_outputs/`: Stores output CSVs for each computed feature.
- `README.md`: Contains sample output logs and PCA weights for reference.

## Major Components & Data Flow
- **OFI_Creation class**: Central class for all feature engineering. Handles data cleaning, feature creation, and PCA integration.
- **Workflow**:
  1. Load and preprocess CSV data (handle canceled orders, drop trades).
  2. Compute OFI features for best level, multi-level (with interval resampling), and integrated (via PCA).
  3. Save results to CSVs in `feature_outputs/`.
  4. Optionally, compute cross-asset OFI features using multiple assets (see `create_cross_asset_ofi_feature`).

## Developer Workflows
- **Run main pipeline**: Execute `main.py` to process the input CSV and generate all feature outputs. No build step required.
- **Debugging**: Print statements in `main.py` show progress and PCA weights. Errors are raised with context.
- **Adding new features**: Extend `OFI_Creation` methods or add new output CSVs in `feature_outputs/`.

## Project-Specific Patterns
- **Level handling**: All order book levels are referenced with two-digit strings (e.g., `bid_px_00`, `ask_sz_09`).
- **Interval resampling**: Multi-level OFI features are resampled to a fixed interval (default: 30s).
- **PCA integration**: Integrated OFI uses PCA on multi-level features, normalizing weights to sum to 1.
- **Cross-asset features**: Use a dictionary of DataFrames keyed by asset name for cross-asset OFI computation.
- **Error handling**: All major computations are wrapped in try/except blocks with informative printouts.

## External Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib (optional, not used in output)

## Conventions & Integration Points
- Input CSV must match expected column names and structure (see `main.py` for details).
- Output CSVs are written to `feature_outputs/` and are overwritten on each run.
- No test suite or build system is present; all logic is in `main.py`.

## Example Usage
```bash
python main.py
```

## References
- See `README.md` for sample output and PCA weights.
- Key logic and conventions are in `main.py`.

---

If any section is unclear or missing, please provide feedback to improve these instructions.
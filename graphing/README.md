# FastGraphCompute Performance Analysis & Visualization

This repository contains tools for processing benchmark data and generating publication-ready performance comparison visualizations for [FastGraphCompute](https://github.com/jkiesele/FastGraphCompute/) (FGC) against state-of-the-art nearest neighbor search algorithms.

## Overview

FastGraphCompute is a high-performance extension for PyTorch designed to accelerate graph-based operations in Graph Neural Networks (GNNs). This plotting repository provides comprehensive performance analysis tools that compare FGC against:

- **FAISS** (Facebook AI Similarity Search)
- **ScaNN** (Scalable Nearest Neighbors)
- **HNSWLIB** (Hierarchical Navigable Small World)
- **Annoy** (Approximate Nearest Neighbors Oh Yeah)
- **GGNN** (Gated Graph Neural Network)

## Repository Structure

```
.
├── process_csvs.py          # Main data processing and plotting script
├── plot_performances.py     # Alternative plotting script
├── combine_csvs.py          # Utility to combine per-run CSV files
├── fix_ggnn.py              # Utility to fix GGNN data issues
├── master_data.csv          # Merged master dataset (generated)
├── plots/                   # Generated visualization outputs
├── faiss-data/              # Raw FAISS benchmark data
├── scann-data/              # Raw ScaNN benchmark data
├── hnswlib-data/            # Raw HNSWLIB benchmark data
├── annoy-data/              # Raw Annoy benchmark data
└── ggnn-data/               # Raw GGNN benchmark data
```

## Installation

### Prerequisites

- Python 3.8+
- pip

### Dependencies

Install required packages:

```bash
pip install pandas numpy plotly kaleido
```

For high-quality image export, `kaleido` is required (installed automatically with plotly).

## Usage

### Main Workflow: `process_csvs.py`

The primary script handles the complete data processing and visualization pipeline. It supports three processing modes:

#### 1. Process Raw Data (`PROCESSING_MODE = "process"`)

Combines CSV files from algorithm directories, applies weighted averaging, and generates cleaned datasets:

```python
# In process_csvs.py, set:
PROCESSING_MODE = "process"
```

This mode:
- Reads all CSV files from algorithm-specific directories (`faiss-data/`, `scann-data/`, etc.)
- Standardizes column names (`dims` → `dimension`, `fixed_dimension` → `dimension`, etc.)
- Applies weighted averaging (weighted by `count`) for duplicate `(size, k, dimension)` combinations
- Generates cleaned CSV files: `*_data_cleaned.csv`

#### 2. Merge Cleaned Data (`PROCESSING_MODE = "merge"`)

Merges all cleaned CSV files into a single master dataset:

```python
# In process_csvs.py, set:
PROCESSING_MODE = "merge"
```

This mode:
- Loads all `*_data_cleaned.csv` files
- Merges them on `(size, k, dimension)` keys
- Creates `master_data.csv` with all algorithm times and FGC times
- Generates all visualization plots

#### 3. Use Existing Data (`PROCESSING_MODE = "use_existing"`)

Skips processing and uses existing cleaned files:

```python
# In process_csvs.py, set:
PROCESSING_MODE = "use_existing"
```

### Running the Main Script

```bash
python process_csvs.py
```

### Generated Visualizations

The script automatically generates the following publication-ready plots in the `plots/` directory:

1. **`fgc_speedup_d3_all_algorithms.png`**
   - Side-by-side comparison (standard + zoomed view)
   - D=3, K=40, varying dataset sizes
   - Compares FGC speedup against all algorithms

2. **`fgc_speedup_d5_all_algorithms.png`**
   - Side-by-side comparison (standard + zoomed view)
   - D=5, K=40, varying dataset sizes
   - Compares FGC speedup against all algorithms

3. **`fgc_k_comparison_1M_d2-10_all_algorithms.png`**
   - Three-panel comparison (K=10, K=40, K=100)
   - 1M vectors, dimensions 2-10
   - Shows how speedup varies with different K values

4. **`fgc_dimensional_scaling_1M_k40_all_algorithms.png`**
   - Side-by-side comparison (standard + zoomed view)
   - 1M vectors, K=40, dimensions up to 15
   - Analyzes dimensional scaling behavior

5. **`fgc_speedup_d3_vs_d5_k40_all_algorithms.png`**
   - Side-by-side comparison of D=3 vs D=5
   - K=40, varying dataset sizes
   - Direct dimensional comparison

### Alternative Plotting Script: `plot_performances.py`

An alternative plotting script with different visualization options:

```bash
python plot_performances.py
```

**Note:** This script uses hardcoded file paths and may require modification for your environment.

### Utility Scripts

#### `combine_csvs.py`

Combines multiple per-run CSV files into single algorithm files:

```bash
python combine_csvs.py
```

This is useful when you have multiple benchmark runs that need to be consolidated before processing.

#### `fix_ggnn.py`

Fixes missing or invalid size values in GGNN data:

```bash
python fix_ggnn.py
```

Sets missing `size` values to 1M (1,000,000) for consistency.

## Data Format

### Input CSV Format

Each algorithm's CSV files should contain the following columns:

- `size`: Dataset size (number of vectors)
- `k`: Number of nearest neighbors to find
- `dimension` (or `dims`): Vector dimensionality
- `{algorithm}_time`: Execution time for the algorithm (e.g., `faiss_time`, `scann_time`)
- `fgc_time`: Execution time for FGC baseline
- `count`: Number of runs (for weighted averaging)

### Output Format

Cleaned CSV files contain:
- Standardized column names
- Weighted-averaged times (duplicate `(size, k, dimension)` combinations merged)
- All required columns for plotting

The master CSV (`master_data.csv`) contains:
- All algorithm time columns: `faiss_time`, `scann_time`, `hnswlib_time`, `annoy_time`, `ggnn_time`
- FGC time columns: `fgc_time` (base) and algorithm-specific variants
- Standardized `size`, `k`, `dimension` columns

## Customization

### Algorithm Configuration

Algorithms are configured in the `ALGORITHMS` dictionary in `process_csvs.py`:

```python
ALGORITHMS = {
    'FAISS': {
        'display_name': 'FAISS',
        'color': '#1B9E77',
        'time_column': 'faiss_time',
        'marker_symbol': 'circle'
    },
    # ... other algorithms
}
```

### Plot Customization

Key parameters in `process_csvs.py`:

- `MAX_DATASET_SIZE`: Maximum dataset size to analyze (default: 5,000,000)
- Plot dimensions, colors, and styling can be modified in individual plotting functions
- Y-axis caps for zoomed views can be adjusted per plot

## Output Specifications

All plots are generated as high-resolution PNG files with:
- **Resolution**: 2× scale factor (e.g., 1400×600 → 2800×1200 pixels)
- **Format**: PNG
- **Style**: Professional, publication-ready with consistent color schemes
- **Font**: Arial, optimized for readability

## Contributing

When adding new algorithms or modifying plots:

1. Add algorithm configuration to `ALGORITHMS` dictionary
2. Ensure data directory follows naming convention: `{algorithm}-data/`
3. Update processing logic if column names differ
4. Test with sample data before processing full datasets

## Related Repository

This plotting repository is designed to work with benchmark data from:
**[FastGraphCompute](https://github.com/jkiesele/FastGraphCompute/)** - The main algorithm repository

## License

This repository is part of the FastGraphCompute research project. Please refer to the main repository for licensing information.

## Citation

If you use these visualization tools in your research, please cite the FastGraphCompute paper (forthcoming) and this repository.

---

**Note**: This repository is actively maintained as part of ongoing research. For questions or issues, please refer to the main FastGraphCompute repository.


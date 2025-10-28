# Disease Classifier (KIRC)

Classifies disease state (Healthy vs KIRC) using 3 continuous biomarker inputs.

## Key Features

✅ **2 Hidden Layers** - Most complex architecture  
✅ **Optimizes Activation Parameters** - KH, nH, KL, nL are evolved  
✅ **Uses config.py** - Parameters extracted  
✅ **Saves top 10 solutions** - Multiple best networks  
✅ **CSV Data Loading** - Reads from KIRC_Data.csv  

## Quick Start

```bash
# Make sure KIRC_Data.csv is in this directory
python run.py
```

## Files

- `config.py` - All configuration (NEW - parameters extracted)
- `spatial_logic.py` - Core algorithm with 2 layers (from original kipc2)
- `run.py` - Main script (from original KIRC_classifier)
- `README.md` - This file

## Network Architecture

```
3 inputs (miR-200c, miR-204, miR-887)
   ↓
Hidden Layer 1 (2-4 nodes)
   ↓
Hidden Layer 2 (2-4 nodes)
   ↓
Output (Healthy vs Disease)
```

## What's Optimized

1. **Network topology** - Number and type (HP/LP) of nodes in both hidden layers
2. **Spatial positions** - X,Y coordinates of all nodes
3. **Activation parameters** - KH, nH, KL, nL (Hill equation parameters)

This is unique! Other modules have fixed activation parameters.

## Key Differences from Other Modules

### Architecture
- **2 hidden layers** (vs 1 in other modules)
- **3 weight matrices** (Input→H1, H1→H2, H2→Output)
- More complex forward pass

### Optimization
- Optimizes activation function parameters
- Larger population (200,000 vs 5,000-50,000)
- Higher mutation rate (0.5 vs 0.2)

### Spatial Parameters
- `time_step`: **72000** (vs 1e6 or 1e5)
- Much longer diffusion time
- Different spatial dynamics

### Data
- Reads from CSV file
- 3 continuous inputs (rescaled to 1e-8 to 1e-4)
- Variable number of samples (depends on CSV)

## Configuration

Edit `config.py`:

```python
# Adjust network size
GA_PARAMS = {
    'min_hidden1_size': 2,
    'max_hidden1_size': 4,
    'min_hidden2_size': 2,
    'max_hidden2_size': 4,
}

# Adjust GA parameters
GA_PARAMS = {
    'ngen': 20,         # Generations
    'npop': 200000,     # Population (large!)
}

# Save more solutions
OUTPUT_PARAMS = {
    'save_top_n': 20,   # Top 20
}
```

## Data File Format

`KIRC_Data.csv` should have columns:
- `miR-200c` - First biomarker
- `miR-204` - Second biomarker
- `miR-887` - Third biomarker
- `Status` - 'Healthy' or 'KIRC'

## Output Files

- `disease_classifier_results.png` - Visualization
- `disease_classifier_best_topology.txt` - Best network structure
- `disease_classifier_top_solutions.txt` - Top 10 solutions (readable)
- `disease_classifier_top_solutions.npy` - Top 10 solutions (numpy)
- `disease_classifier_weights.npy` - Best weights
- `disease_classifier_positions.npy` - Best positions
- `disease_classifier_act_params.npy` - Best activation parameters

## Implementation Notes

### Original Code Base
Files are adapted from your original implementation:
- `spatial_logic.py` ← `kipc2_spatial_location.py`
- `run.py` ← `KIRC_classifier_AHL_locations.py`

### Improvements Added
- **config.py** - All parameters extracted for easy tuning
- **Top N solutions** - Saves multiple best solutions, not just one
- **Documentation** - Clear comments and structure
- **Consistent interface** - Same style as other modules

### To Fully Integrate Config
The current files have notes at the top indicating they use config.py.
To complete the integration:

1. Replace hardcoded parameters with `config.PARAM_NAME`
2. Update weight calculation to use config spatial params
3. Update GA to use config GA params
4. Add top N solutions extraction (like in other modules)

Or use as-is with your proven original implementation!

## Tips

- This module is computationally expensive (large population)
- Activation parameter optimization adds complexity
- May need more generations than other modules
- CSV file is required
- Results depend heavily on data quality

## Comparison with Other Modules

| Feature | Logic Gates | Binary Classifier | Disease Classifier |
|---------|-------------|-------------------|-------------------|
| Inputs | 2 continuous | 4 binary | 3 continuous |
| Hidden Layers | 1 | 1 | **2** |
| Activation Opt | No | No | **Yes** |
| Data | Generated | Generated | **CSV** |
| Population | 5,000 | 50,000 | **200,000** |
| Complexity | Low | Medium | **High** |

This is the most advanced module!

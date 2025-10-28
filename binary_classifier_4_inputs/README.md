# Binary Classifier with 4 Inputs

Classifies numbers 0-9 using 4 binary inputs (each number encoded as 4-bit binary).

## Operations

- **Prime**: Detects prime numbers (2, 3, 5, 7)
- **PerfectPower**: Detects perfect powers (0, 1, 4, 8, 9)
- **Vowel**: Detects vowels when mapped to letters A-J (A=0, E=4, I=8)

## Quick Start

```bash
# Run single operation
python run.py --ops Prime

# Run multiple operations
python run.py --ops Prime PerfectPower

# Run all operations
python run.py --all

# View configuration
python run.py --config-summary
```

## Key Features

✅ **Uses config.py** - All parameters extracted from code  
✅ **Saves top 10 solutions** - Not just the best one  
✅ **4 binary inputs** - Each number 0-9 encoded as 4 bits  
✅ **Clean code structure** - Well-organized and documented  

## Files

- `config.py` - All configuration parameters
- `data_generator.py` - Binary encoding and operation functions
- `spatial_logic.py` - Core GA and network with 4 inputs
- `run.py` - Main execution script
- `README.md` - This file

## Differences from Logic Gates Module

### Data
- **Fixed 10 samples** (0-9) vs variable samples
- **4 binary inputs** vs 2 continuous inputs
- Input encoding: HIGH/LOW cutoffs

### Spatial Parameters
- `time_step`: **1e5** (vs 1e6 in logic gates)
- `spatial_scale`: **1e-6** (vs 1e-5 in logic gates)
- Larger position ranges (up to 20000 vs 10000)

### Network Architecture
- **4 inputs** → hidden → output
- Weight matrix: `wH.reshape(n_hidden, 4)`

## Output Files

Each run generates:
- `binary_classifier_results_[OPS].png` - Bar plot visualization
- `binary_classifier_best_topology_[OPS].txt` - Best topologies
- `binary_classifier_top_solutions_[OPS].txt` - Top 10 solutions (readable)
- `binary_classifier_top_solutions_[OPS].npy` - Top 10 solutions (numpy)
- `binary_classifier_weights_[OP].npy` - Best weights per operation
- `binary_classifier_positions_[OP].npy` - Best positions per operation
- `binary_classifier_predictions_[OP].npy` - Predictions per operation

## Configuration

Edit `config.py` to adjust:

```python
# For faster testing
GA_PARAMS = {
    'ngen': 20,      # Fewer generations
    'npop': 10000,   # Smaller population
}

# For quality results
GA_PARAMS = {
    'ngen': 50,      # More generations
    'npop': 50000,   # Larger population
}

# Save more top solutions
OUTPUT_PARAMS = {
    'save_top_n': 20,  # Save top 20
}
```

## Example Output

```
Running operations: ['Prime']

======================================================================
Processing Prime
======================================================================
Starting optimization for Prime
Generations: 50, Population: 50000
Generation 0: Best fitness = 0.123456
Generation 10: Best fitness = 0.098765
...
Optimization complete!
Final best fitness: 0.012345
Best topology: ['HP', 'LP', 'HP', 'LP']
Top 10 solutions extracted

Completed Prime: ['HP', 'LP', 'HP', 'LP']
Predictions: [off_cutoff, off_cutoff, on_cutoff, on_cutoff, ...]
Expected:    [off_cutoff, off_cutoff, on_cutoff, on_cutoff, ...]
```

## Tips

- Prime is usually easiest to optimize
- PerfectPower may need more generations
- Vowel has only 3 positive cases (0, 4, 8)
- Increase population size for better results
- All parameters in config.py can be tuned

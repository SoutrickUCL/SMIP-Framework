# Logic Gates with 2 Inputs

This module optimizes neural networks to implement 2-input logic gates (OR, AND, NOR, NAND, XOR) using genetic algorithms with spatial positioning.

## Quick Start

```bash
# Run single gate
python run.py --gates OR

# Run multiple gates
python run.py --gates OR AND XOR

# Run all gates
python run.py --all

# Interactive selection
python run.py --interactive

# View configuration
python run.py --config-summary
```

## Files

- **config.py**: All configuration parameters
- **spatial_logic.py**: Core optimization algorithm
- **data_generator.py**: Logic gate data generation functions
- **run.py**: Main execution script

## Configuration

Edit `config.py` to adjust:

### Training Parameters
```python
DATA_PARAMS = {
    'default_ndata': 20000,    # Training samples (1000 for testing, 20000 for quality)
    'default_ntest': 50000,    # Test samples
}
```

### Genetic Algorithm
```python
GA_PARAMS = {
    'ngen': 10,                # Generations (10 for fast, 25+ for quality)
    'npop': 5000,              # Population (5000 for fast, 20000+ for quality)
}
```

### Spatial Layout
```python
SPATIAL_PARAMS = {
    'input_nodes_x_range': (1000, 8000),
    'input_nodes_y_range': (1000, 8000),
    'output_node_x_range': (10000, 10000),  # Fixed output position
    'output_node_y_range': (5000, 5000),
    'min_distance': 2000,
}
```

### Top Solutions
```python
OUTPUT_PARAMS = {
    'save_top_n': 10,          # Number of top solutions to save
}
```

## Output Files

For each run, the following files are generated:

1. **Plots**:
   - `logic_gate_training_[GATES].png` - Training data visualization
   - `logic_gate_prediction_[GATES].png` - Prediction visualization
   - `logic_gate_truth_table_[GATES].png` - Truth table bar plots

2. **Best Solution**:
   - `logic_gate_weights_[GATE].npy` - Optimized weights
   - `logic_gate_positions_[GATE].npy` - Node positions
   - `logic_gate_data_[GATE].npy` - Test data and predictions
   - `logic_gate_best_topology_[GATES].txt` - Best network topology

3. **Top N Solutions**:
   - `logic_gate_top_solutions_[GATES].txt` - Human-readable format
   - `logic_gate_top_solutions_[GATES].npy` - Numpy format with all details

## How It Works

1. **Spatial Neural Network**: Nodes are positioned in 2D space, and connection weights are calculated using a diffusion model
2. **Genetic Algorithm**: Simultaneously optimizes network topology and spatial positions
3. **Logic Gates**: Networks learn to map continuous inputs to discrete logic outputs

## Available Gates

- **OR**: Output HIGH if either input is HIGH
- **NOR**: Output HIGH only if both inputs are LOW
- **AND**: Output HIGH only if both inputs are HIGH
- **NAND**: Output LOW only if both inputs are HIGH
- **XOR**: Output HIGH if inputs are different

## Tips

- Start with OR gate (easiest to optimize)
- Use small parameters for initial testing (ndata=1000, ngen=10)
- XOR is the hardest - use highest quality settings
- Adjust spatial ranges for different network layouts

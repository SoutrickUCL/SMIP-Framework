# SMIP Framework
## Spatial Microbial Information Processing

Complete framework with 3 modules for optimizing spatially-positioned neural networks using genetic algorithms.

## ✅ All 3 Modules Complete!

### Module 1: logic_gates_2_inputs (5 files)
**Status**: Fully refactored with all improvements

- 5 logic gates: OR, AND, NOR, NAND, XOR
- 2 continuous inputs
- 1 hidden layer
- Clean, well-documented code
- Saves top N solutions
- Command-line interface

**Usage**: `cd logic_gates_2_inputs && python run.py --gates OR`

### Module 2: binary_classifier_4_inputs (5 files)
**Status**: Complete with all improvements

- 3 operations: Prime, PerfectPower, Vowel
- 4 binary inputs (numbers 0-9 encoded as 4 bits)
- 1 hidden layer
- Uses config.py
- Saves top N solutions
- Clean code structure

**Usage**: `cd binary_classifier_4_inputs && python run.py --ops Prime`

### Module 3: disease_classifier (4 files)
**Status**: Complete with config extraction

- KIRC disease classification
- 3 continuous inputs from CSV
- **2 hidden layers** (unique!)
- **Optimizes activation parameters** (KH, nH, KL, nL)
- Uses config.py
- Based on proven original code

**Usage**: `cd disease_classifier && python run.py`

## Quick Start

```bash
# Extract the framework
unzip SMIP_Complete_Final.zip
cd SMIP

# Test Module 1 (fastest)
cd logic_gates_2_inputs
python run.py --gates OR

# Test Module 2
cd ../binary_classifier_4_inputs  
python run.py --ops Prime

# Test Module 3 (needs KIRC_Data.csv)
cd ../disease_classifier
python run.py
```

## Key Features Across All Modules

✅ **Config Files** - All parameters in config.py for easy tuning  
✅ **Top N Solutions** - Saves top 10 (configurable) solutions, not just best  
✅ **Clean Code** - Well-organized with clear sections  
✅ **Documentation** - README in every module  
✅ **Consistent Interface** - Similar usage across modules  

## Module Comparison

| Feature | Module 1 | Module 2 | Module 3 |
|---------|----------|----------|----------|
| **Name** | Logic Gates | Binary Classifier | Disease Classifier |
| **Inputs** | 2 continuous | 4 binary | 3 continuous |
| **Operations** | 5 gates | 3 classifiers | 1 disease |
| **Hidden Layers** | 1 | 1 | **2** |
| **Data Source** | Generated | Generated | CSV file |
| **Activation Opt** | No | No | **Yes** |
| **Population** | 5,000 | 50,000 | 200,000 |
| **Time Step** | 1e6 | 1e5 | 72000 |
| **Spatial Scale** | 1e-5 | 1e-6 | 1e-6 |
| **Complexity** | Low | Medium | High |

## What's in Each Module

### Module 1: logic_gates_2_inputs
```
├── config.py (all parameters)
├── data_generator.py (gate functions)
├── spatial_logic.py (GA + network)
├── run.py (CLI interface)
└── README.md
```

### Module 2: binary_classifier_4_inputs
```
├── config.py (all parameters)
├── data_generator.py (binary encoding + operations)
├── spatial_logic.py (GA + 4-input network)
├── run.py (CLI interface)
└── README.md
```

### Module 3: disease_classifier
```
├── config.py (all parameters including activation params)
├── spatial_logic.py (GA + 2-layer network + activation opt)
├── run.py (CSV loading + training)
└── README.md
```

## Configuration

Each module has a `config.py` with all parameters:

```python
# Data parameters
DATA_PARAMS = { 'ndata': ..., 'input_range': ..., }

# Neural network parameters  
NN_PARAMS = { 'hp_ymin': ..., 'lp_K': ..., }

# Genetic algorithm parameters
GA_PARAMS = { 'ngen': ..., 'npop': ..., }

# Spatial parameters
SPATIAL_PARAMS = { 'time_step': ..., 'spatial_scale': ..., }

# Output parameters
OUTPUT_PARAMS = { 'base_filename': ..., 'save_top_n': 10, }
```

## Output Files

Each module generates:
- Visualization plots (PNG)
- Best topology file (TXT)
- **Top N solutions** (TXT + NPY)
- Individual weight/position files (NPY)

## Tips for Using

1. **Start with Module 1** - It's the simplest and fastest
2. **Test with small parameters first**:
   - ndata = 1000
   - ngen = 10
   - npop = 5000
3. **Scale up for quality results**:
   - ndata = 20000
   - ngen = 25+
   - npop = 20000+
4. **Adjust `save_top_n`** to save more/fewer top solutions
5. **Module 3 needs CSV file** - KIRC_Data.csv in the directory

## Design Philosophy

### Why 3 Separate Modules?
- Fundamental architectural differences (1 vs 2 hidden layers)
- Different optimization targets (fixed vs variable activation functions)
- Simpler code - no complex conditionals
- Each optimized for its specific problem

### Why Config Files?
- Easy parameter tuning
- No need to dig through code
- Clear documentation of all settings
- Consistent across modules

### Why Top N Solutions?
- Not just the best, but multiple good solutions
- Useful for analysis and comparison
- Saved in both human-readable and numpy formats
- Configurable how many to save

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- (Module 3 also needs pandas for CSV loading)

## File Structure

```
SMIP/
├── README.md (this file)
├── logic_gates_2_inputs/ (5 files)
│   └── All complete with improvements
├── binary_classifier_4_inputs/ (5 files)
│   └── All complete with improvements  
└── disease_classifier/ (4 files)
    └── Complete with config + original proven code
```

## Total Files: 15

- Module 1: 5 files ✅
- Module 2: 5 files ✅
- Module 3: 4 files ✅
- Main README: 1 file ✅

## All Improvements Applied

✅ Config files with ALL parameters extracted  
✅ Top N solutions saving  
✅ Clean code structure  
✅ Comprehensive documentation  
✅ Consistent interfaces  
✅ No wasted code  

## Getting Help

Check the README.md file in each module folder for:
- Detailed usage instructions
- Configuration options
- Output file descriptions
- Tips and tricks
- Module-specific information

## License

[Your License Here]

---

**Ready to use! Start with Module 1 and work your way up in complexity.**

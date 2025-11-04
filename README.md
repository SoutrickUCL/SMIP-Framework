# SMIP Framework
## Spatial Microbial Information Processing

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Academic-green.svg)](LICENSE)

Neural network optimization for engineered bacterial communities using genetic algorithms and spatial positioning.

**Paper:** *"Designing neural network computation across engineered bacterial communities"*  
**Authors:** Soutrick Das and Chris Barnes  
**Institution:** Department of Cell and Developmental Biology, University College London

---

## Overview

Optimizes neural networks where nodes communicate via diffusion, modeling computation in bacterial communities. Uses genetic algorithms to evolve both network topology and spatial positions.

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/SMIP-Framework.git
cd SMIP-Framework
pip install numpy matplotlib pandas

# Run a test (2-5 minutes)
cd logic_gates_2_inputs
python run.py --gates OR
```

---

## Modules

| Module | Complexity | Layers | Description | Usage |
|--------|-----------|---------|-------------|-------|
| **1. Logic Gates** | Low | 1 | 4 gates (OR, AND, NOR, NAND), 2 inputs | `python run.py --gates OR` |
| **2. Binary Classifier** | Medium | 1 | Prime/PerfectPower/Vowel, 4 binary inputs | `python run.py --ops Prime` |
| **3. Disease Classifier** | High | 2 | KIRC classification, 3 biomarkers, optimizes activation params | `python run.py` |

Each module has its own `README.md` with detailed documentation.

---

## Key Features

- **Config-based**: All parameters in `config.py` files
- **Top N solutions**: Saves multiple best solutions (default: 10)
- **Spatial weights**: Based on diffusion model with Hill equation activations
- **Genetic algorithm**: Evolves topology + spatial positions

---

## Output Files

Each run generates:
- Visualization plots (`.png`)
- Best network topology (`.txt`)
- Top N solutions (`.txt` + `.npy`)
- Individual weights/positions (`.npy`)

---

## Citation

If you use this code, please cite:

```bibtex
@article{das2024designing,
  title={Designing neural network computation across engineered bacterial communities},
  author={Das, Soutrick and Barnes, Chris},
  institution={University College London},
  year={2025}
}
```

---

## License

Academic and research use only. For commercial use inquiries, please contact the authors.

---

## Contact

- **Chris Barnes**: christopher.barnes@ucl.ac.uk  
- **Soutrick Das**: soutrick.das@ucl.ac.uk

Department of Cell and Developmental Biology, University College London

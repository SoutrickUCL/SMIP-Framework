# ==============================================================================
# CONFIG.PY - Configuration for Disease Classifier (KIRC)
# ==============================================================================

class Config:
    """Configuration for disease classifier with 2 hidden layers"""
    
    # ==========================================================================
    #                       DATA PARAMETERS
    # ==========================================================================
    DATA_PARAMS = {
        'csv_file': 'KIRC_Data.csv',
        'input_columns': ['miR-200c', 'miR-204', 'miR-887'],
        'output_column': 'Status',
        
        # Input data will be rescaled to this range
        'rescale_input_min': 1e-8,
        'rescale_input_max': 1e-4,
        
        # Classification thresholds
        'healthy_threshold': 10**2.57,
        'disease_threshold': 10**6.13,
    }
    
    # ==========================================================================
    #                       NEURAL NETWORK PARAMETERS
    # ==========================================================================
    NN_PARAMS = {
        # Initial activation function parameters (THESE WILL BE OPTIMIZED!)
        'hp_K_init': 5e-6,
        'hp_n_init': 2.0,
        'lp_K_init': 5e-6,
        'lp_n_init': 2.0,
        
        # Fixed activation parameters
        'hp_ymin': 2.57,
        'hp_ymax': 6.13,
        'lp_ymin': 3.05,
        'lp_ymax': 5.37,
        
        # Rescaling for inter-layer communication
        'rescale_min': 1e-8,
        'rescale_max': 1e-4,
        
        'activation_functions': ['HP', 'LP']
    }
    
    # ==========================================================================
    #                       GENETIC ALGORITHM PARAMETERS
    # ==========================================================================
    GA_PARAMS = {
        'ngen': 20,                      # Number of generations
        'npop': 200000,                  # Large population (more complex problem)
        'mutation_prob': 0.5,            # Higher mutation rate
        'pop_cut_ratio': 0.5,            # Keep top 50%
        
        # Network size ranges for BOTH hidden layers
        'min_hidden1_size': 2,
        'max_hidden1_size': 4,
        'min_hidden2_size': 2,
        'max_hidden2_size': 4,
        
        'segment_ratio': 1/6,
        'segment_pos_ratio': 1/3,
        'recombination_ratio': 1/5,
        
        # Activation parameter mutation ranges
        'K_mutation_factor': 0.1,        # ±10% of current value
        'n_mutation_range': 0.2,         # ±0.2 from current value
    }
    
    # ==========================================================================
    #                       SPATIAL PARAMETERS
    # ==========================================================================
    SPATIAL_PARAMS = {
        # Physical parameters (DIFFERENT from other modules!)
        'diff_coefficient': 1e-11,
        'time_step': 72000,              # MUCH LONGER than other modules
        'spatial_scale': 1e-6,
        
        'min_distance': 2000,
        
        # Position ranges (larger than logic gates)
        'input_nodes_x_range': (500, 8000),
        'input_nodes_y_range': (500, 8000),
        'hidden_nodes_x_range': (500, 8000),
        'hidden_nodes_y_range': (500, 8000),
        'output_node_x_range': (10000, 10000),
        'output_node_y_range': (5000, 5000),
        'new_node_x_range': (500, 7500),
        'new_node_y_range': (500, 7500)
    }
    
    # ==========================================================================
    #                       OUTPUT PARAMETERS
    # ==========================================================================
    OUTPUT_PARAMS = {
        'base_filename': 'disease_classifier',
        'save_top_n': 10,
    }
    
    # ==========================================================================
    #                       HELPER METHODS
    # ==========================================================================
    
    def print_config_summary(self):
        """Print configuration summary"""
        print("=" * 70)
        print("DISEASE CLASSIFIER (KIRC) - CONFIGURATION")
        print("=" * 70)
        print(f"Data file: {self.DATA_PARAMS['csv_file']}")
        print(f"Input columns: {', '.join(self.DATA_PARAMS['input_columns'])}")
        print(f"Output column: {self.DATA_PARAMS['output_column']}")
        print()
        print(f"Network: 3 inputs → hidden1 → hidden2 → output (2 HIDDEN LAYERS!)")
        print(f"Hidden1 size: {self.GA_PARAMS['min_hidden1_size']}-{self.GA_PARAMS['max_hidden1_size']}")
        print(f"Hidden2 size: {self.GA_PARAMS['min_hidden2_size']}-{self.GA_PARAMS['max_hidden2_size']}")
        print()
        print(f"GA generations: {self.GA_PARAMS['ngen']}")
        print(f"GA population: {self.GA_PARAMS['npop']} (LARGE!)")
        print()
        print("Spatial parameters (NOTE: Very different from other modules!):")
        print(f"  Time step: {self.SPATIAL_PARAMS['time_step']} (vs 1e6 or 1e5)")
        print(f"  Spatial scale: {self.SPATIAL_PARAMS['spatial_scale']}")
        print()
        print("OPTIMIZABLE PARAMETERS (unique to this module):")
        print(f"  hp_K, hp_n, lp_K, lp_n (activation function parameters)")
        print()
        print(f"Top solutions to save: {self.OUTPUT_PARAMS['save_top_n']}")
        print("=" * 70)

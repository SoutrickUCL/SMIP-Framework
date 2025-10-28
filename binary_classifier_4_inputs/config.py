# ==============================================================================
# CONFIG.PY - Configuration for Binary Classifier with 4 Inputs
# ==============================================================================

class Config:
    """Configuration settings for 4-input binary classifier"""
    
    # ==========================================================================
    #                       AVAILABLE OPERATIONS
    # ==========================================================================
    AVAILABLE_OPERATIONS = ['Prime', 'PerfectPower', 'Vowel']
    
    # ==========================================================================
    #                       DATA GENERATION PARAMETERS
    # ==========================================================================
    DATA_PARAMS = {
        'ndata': 10,                     # Fixed: 10 samples (0-9)
        'on_cutoff': 10**6.13,           # HIGH value for binary encoding
        'off_cutoff': 10**2.57,          # LOW value for binary encoding
    }
    
    # ==========================================================================
    #                       NEURAL NETWORK PARAMETERS
    # ==========================================================================
    NN_PARAMS = {
        # Output thresholds
        'on_cutoff': 10**6.13,
        'off_cutoff': 10**2.57,
        
        # High Pass (HP) activation function
        'hp_ymin': 2.57,
        'hp_ymax': 6.13,
        'hp_K': 6.98e-9,
        'hp_n': 0.866,
        
        # Low Pass (LP) activation function
        'lp_ymin': 3.05,
        'lp_ymax': 5.37,
        'lp_K': 7.26e-8,
        'lp_n': 2.11,
        
        # Rescaling parameters
        'rescale_target_min': 1e-15,
        'rescale_target_max': 1e-2,
        
        'activation_functions': ['HP', 'LP']
    }
    
    # ==========================================================================
    #                       GENETIC ALGORITHM PARAMETERS
    # ==========================================================================
    GA_PARAMS = {
        'ngen': 50,                      # Number of generations
        'npop': 50000,                   # Population size
        'mutation_prob': 0.2,            # Mutation probability
        'pop_cut_ratio': 0.5,            # Elitism ratio (keep top 50%)
        'min_network_size': 3,           # Minimum hidden nodes
        'max_network_size': 5,           # Maximum hidden nodes
        'segment_ratio': 1/6,            # Mutation segment size
        'segment_pos_ratio': 1/3,        # Position mutation size
        'recombination_ratio': 1/5       # Recombination rate
    }
    
    # ==========================================================================
    #                       SPATIAL PARAMETERS
    # ==========================================================================
    SPATIAL_PARAMS = {
        # Physical parameters (DIFFERENT from logic gates!)
        'diff_coefficient': 1e-11,       # Diffusion coefficient (m²/s)
        'time_step': 1e5,                # Time step (SHORTER than logic gates)
        'spatial_scale': 1e-6,           # Spatial scale (SMALLER than logic gates)
        
        # Minimum distance between nodes
        'min_distance': 2000,
        
        # Node position ranges (LARGER than logic gates)
        'input_nodes_x_range': (1000, 16000),
        'input_nodes_y_range': (1000, 16000),
        'hidden_nodes_x_range': (1000, 16000),
        'hidden_nodes_y_range': (1000, 16000),
        'output_node_x_range': (20000, 20000),   # Fixed at 20000
        'output_node_y_range': (10000, 10000),   # Fixed at 10000
        'new_node_x_range': (1000, 15000),
        'new_node_y_range': (1000, 15000)
    }
    
    # ==========================================================================
    #                       OUTPUT PARAMETERS
    # ==========================================================================
    OUTPUT_PARAMS = {
        'base_filename': 'binary_classifier',
        'save_top_n': 10,                # Number of top solutions to save
    }
    
    # ==========================================================================
    #                       HELPER METHODS
    # ==========================================================================
    
    def print_config_summary(self):
        """Print configuration summary"""
        print("=" * 70)
        print("BINARY CLASSIFIER (4 INPUTS) - CONFIGURATION")
        print("=" * 70)
        print(f"Operations: {', '.join(self.AVAILABLE_OPERATIONS)}")
        print(f"Data samples: {self.DATA_PARAMS['ndata']} (fixed 0-9)")
        print(f"GA generations: {self.GA_PARAMS['ngen']}")
        print(f"GA population: {self.GA_PARAMS['npop']}")
        print(f"Network size: {self.GA_PARAMS['min_network_size']}-{self.GA_PARAMS['max_network_size']} nodes")
        print()
        print("Spatial parameters (NOTE: Different from logic gates!):")
        print(f"  Time step: {self.SPATIAL_PARAMS['time_step']} (vs 1e6 in logic gates)")
        print(f"  Spatial scale: {self.SPATIAL_PARAMS['spatial_scale']} (vs 1e-5 in logic gates)")
        print(f"  Position ranges: up to {self.SPATIAL_PARAMS['input_nodes_x_range'][1]}")
        print(f"  Output position: ({self.SPATIAL_PARAMS['output_node_x_range'][0]}, "
              f"{self.SPATIAL_PARAMS['output_node_y_range'][0]})")
        print()
        print(f"Top solutions to save: {self.OUTPUT_PARAMS['save_top_n']}")
        print("=" * 70)

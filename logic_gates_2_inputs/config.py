# ==============================================================================
# CONFIG.PY - Configuration for Logic Gates with 2 Inputs
# ==============================================================================
# This file contains all configuration parameters for training neural networks
# to implement 2-input logic gates (OR, AND, NOR, NAND, XOR) using genetic
# algorithms with spatial positioning.
# ==============================================================================

class Config:
    """Configuration settings for 2-input logic gate optimization"""
    
    # ==========================================================================
    #                       AVAILABLE LOGIC GATES
    # ==========================================================================
    # List of all supported 2-input logic gates
    AVAILABLE_GATES = ['OR', 'NOR', 'AND', 'NAND', 'XOR']
    
    # ==========================================================================
    #                       DATA GENERATION PARAMETERS
    # ==========================================================================
    DATA_PARAMS = {
        'default_ndata': 20000,          # Number of training data points
                                         # - Use 1000 for quick testing
                                         # - Use 20000 for quality results
        
        'default_ntest': 50000,          # Number of test data points
        
        'input_range': (-15, -2),        # Range for input generation (log10 scale)
                                         # Inputs will be: 10^(-15) to 10^(-2)
        
        'input_threshold': 1e-8,         # Threshold to distinguish 0 vs 1
                                         # Values below this are considered "0"
                                         # Values above are considered "1"
    }
    
    # ==========================================================================
    #                       NEURAL NETWORK PARAMETERS
    # ==========================================================================
    NN_PARAMS = {
        # ----------------------------------------------------------------------
        # Output Classification Thresholds
        # ----------------------------------------------------------------------
        'on_cutoff': 10**6.13,           # High output threshold (logic "1")
        'off_cutoff': 10**2.57,          # Low output threshold (logic "0")
        
        # ----------------------------------------------------------------------
        # High Pass (HP) Activation Function Parameters
        # ----------------------------------------------------------------------
        # HP function: ymin + (ymax - ymin) * (x^n) / (K^n + x^n)
        'hp_ymin': 2.57,                 # HP minimum output (log10)
        'hp_ymax': 6.13,                 # HP maximum output (log10)
        'hp_K': 3.53e-9,                 # HP half-saturation constant
        'hp_n': 2.532,                   # HP Hill coefficient (cooperativity)
        
        # ----------------------------------------------------------------------
        # Low Pass (LP) Activation Function Parameters
        # ----------------------------------------------------------------------
        # LP function: ymin + (ymax - ymin) * (K^n) / (K^n + x^n)
        'lp_ymin': 3.05,                 # LP minimum output (log10)
        'lp_ymax': 5.37,                 # LP maximum output (log10)
        'lp_K': 7.26e-8,                 # LP half-saturation constant
        'lp_n': 2.11,                    # LP Hill coefficient
        
        # ----------------------------------------------------------------------
        # Rescaling Parameters (for intermediate layers)
        # ----------------------------------------------------------------------
        'rescale_target_min': 1e-15,     # Target minimum after rescaling
        'rescale_target_max': 1e-2,      # Target maximum after rescaling
        
        # ----------------------------------------------------------------------
        # Available Activation Functions
        # ----------------------------------------------------------------------
        'activation_functions': ['HP', 'LP']  # High-pass and Low-pass
    }
    
    # ==========================================================================
    #                       GENETIC ALGORITHM PARAMETERS
    # ==========================================================================
    GA_PARAMS = {
        'ngen': 10,                      # Number of generations
                                         # - Use 10 for fast testing
                                         # - Use 25+ for quality results
        
        'npop': 5000,                    # Population size
                                         # - Use 5000 for fast testing
                                         # - Use 20000+ for quality results
        
        'mutation_prob': 0.2,            # Probability of topology mutation
                                         # vs position mutation
        
        'pop_cut_ratio': 0.5,            # Fraction of population to keep
                                         # (elitism - keeps best performers)
        
        'min_network_size': 3,           # Minimum number of hidden nodes
        'max_network_size': 4,           # Maximum number of hidden nodes
        
        'segment_ratio': 1/6,            # Fraction of individuals to mutate
                                         # in each mutation segment
        
        'segment_pos_ratio': 1/3,        # Fraction of positions to mutate
                                         # in position mutation
        
        'recombination_ratio': 1/5       # Fraction of population to
                                         # recombine in each generation
    }
    
    # ==========================================================================
    #                       SPATIAL PARAMETERS
    # ==========================================================================
    # These parameters control the spatial layout of nodes and how spatial
    # distance affects connection weights through diffusion physics
    # ==========================================================================
    SPATIAL_PARAMS = {
        # ----------------------------------------------------------------------
        # Physical Diffusion Parameters
        # ----------------------------------------------------------------------
        'diff_coefficient': 1e-11,       # Diffusion coefficient (m²/s)
                                         # Models molecular diffusion
        
        'time_step': 1e6,                # Time step for diffusion (seconds)
                                         # ~11.6 days
        
        'spatial_scale': 1e-5,           # Spatial scaling factor (meters/unit)
                                         # Converts grid units to physical distance
        
        # ----------------------------------------------------------------------
        # Minimum Distance Constraint
        # ----------------------------------------------------------------------
        'min_distance': 2000,            # Minimum distance between any two nodes
                                         # Prevents nodes from overlapping
        
        # ----------------------------------------------------------------------
        # Node Position Ranges (in grid units)
        # ----------------------------------------------------------------------
        # Input nodes can be placed in this range
        'input_nodes_x_range': (1000, 8000),
        'input_nodes_y_range': (1000, 8000),
        
        # Hidden layer nodes can be placed in this range
        'hidden_nodes_x_range': (1000, 8000),
        'hidden_nodes_y_range': (1000, 8000),
        
        # Output node has FIXED position
        'output_node_x_range': (10000, 10000),   # Fixed X coordinate
        'output_node_y_range': (5000, 5000),     # Fixed Y coordinate
        
        # New nodes added during mutation use this range
        'new_node_x_range': (500, 7000),
        'new_node_y_range': (500, 7000)
    }
    
    # ==========================================================================
    #                       OUTPUT PARAMETERS
    # ==========================================================================
    OUTPUT_PARAMS = {
        'base_filename': 'logic_gate',   # Base name for all output files
        'save_top_n': 10,                # Number of top solutions to save
    }
    
    # ==========================================================================
    #                       ANALYSIS PARAMETERS
    # ==========================================================================
    # Test inputs for evaluating trained logic gates
    # Format: [input1, input2] where values represent logic 0 or 1
    # ==========================================================================
    ANALYSIS_PARAMS = {
        'test_inputs': [
            [1e-15, 1e-15],              # [0, 0]
            [1e-2, 1e-15],               # [1, 0]
            [1e-15, 1e-2],               # [0, 1]
            [1e-2, 1e-2]                 # [1, 1]
        ],
    }
    
    # ==========================================================================
    #                       HELPER METHODS
    # ==========================================================================
    
    def print_config_summary(self):
        """Print a summary of current configuration"""
        print("=" * 70)
        print("LOGIC GATES (2 INPUTS) - CONFIGURATION SUMMARY")
        print("=" * 70)
        print(f"Training data points:  {self.DATA_PARAMS['default_ndata']}")
        print(f"Test data points:      {self.DATA_PARAMS['default_ntest']}")
        print(f"GA generations:        {self.GA_PARAMS['ngen']}")
        print(f"GA population:         {self.GA_PARAMS['npop']}")
        print(f"Network size range:    {self.GA_PARAMS['min_network_size']}-{self.GA_PARAMS['max_network_size']} nodes")
        print()
        print(f"Input X range:         {self.SPATIAL_PARAMS['input_nodes_x_range']}")
        print(f"Input Y range:         {self.SPATIAL_PARAMS['input_nodes_y_range']}")
        print(f"Hidden X range:        {self.SPATIAL_PARAMS['hidden_nodes_x_range']}")
        print(f"Hidden Y range:        {self.SPATIAL_PARAMS['hidden_nodes_y_range']}")
        print(f"Output position:       ({self.SPATIAL_PARAMS['output_node_x_range'][0]}, "
              f"{self.SPATIAL_PARAMS['output_node_y_range'][0]})")
        print(f"Minimum distance:      {self.SPATIAL_PARAMS['min_distance']}")
        print()
        print(f"Diffusion coefficient: {self.SPATIAL_PARAMS['diff_coefficient']}")
        print(f"Time step:             {self.SPATIAL_PARAMS['time_step']}")
        print(f"Spatial scale:         {self.SPATIAL_PARAMS['spatial_scale']}")
        print()
        print(f"On/Off cutoffs:        {self.NN_PARAMS['on_cutoff']:.2e} / "
              f"{self.NN_PARAMS['off_cutoff']:.2e}")
        print("=" * 70)

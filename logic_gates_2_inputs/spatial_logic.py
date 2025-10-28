# ==============================================================================
# SPATIAL_LOGIC.PY - Core Algorithm for 2-Input Logic Gates
# ==============================================================================
# This file contains the neural network, spatial weight calculation, and
# genetic algorithm implementation for optimizing logic gate networks.
# ==============================================================================

import numpy as np
import math
from config import Config

# Load configuration
config = Config()

# ==============================================================================
#                       ACTIVATION FUNCTIONS
# ==============================================================================
# High-pass (HP) and Low-pass (LP) response functions based on Hill equations
# ==============================================================================

def response_hp(x, ymin, ymax, K, n):
    """
    High-pass response function (increases with input)
    Formula: 10^(ymin + (ymax-ymin) * (x^n)/(K^n + x^n))
    """
    return 10**( ymin + (ymax-ymin)*( (x**n)/( K**n + x**n) ) )


def response_lp(x, ymin, ymax, K, n):
    """
    Low-pass response function (decreases with input)
    Formula: 10^(ymin + (ymax-ymin) * (K^n)/(K^n + x^n))
    """
    return 10**( ymin + (ymax-ymin)*( K**n /( K**n + x**n) ) )


def response_hp_configured(x):
    """High-pass function with parameters from config"""
    return response_hp(x, 
                     config.NN_PARAMS['hp_ymin'],
                     config.NN_PARAMS['hp_ymax'], 
                     config.NN_PARAMS['hp_K'],
                     config.NN_PARAMS['hp_n'])


def response_lp_configured(x):
    """Low-pass function with parameters from config"""
    return response_lp(x,
                     config.NN_PARAMS['lp_ymin'],
                     config.NN_PARAMS['lp_ymax'],
                     config.NN_PARAMS['lp_K'], 
                     config.NN_PARAMS['lp_n'])


# ==============================================================================
#                       RESCALING FUNCTIONS
# ==============================================================================
# Rescale hidden layer outputs to target range for next layer
# ==============================================================================

def rescale_node_HP(x):
    """Rescale high-pass output to target range"""
    src_min = 10**config.NN_PARAMS['hp_ymin']
    src_max = 10**config.NN_PARAMS['hp_ymax']
    target_min = config.NN_PARAMS['rescale_target_min']
    target_max = config.NN_PARAMS['rescale_target_max']
    return target_min + (x - src_min) * (target_max - target_min) / (src_max - src_min)


def rescale_node_LP(x):
    """Rescale low-pass output to target range"""
    src_min = 10**config.NN_PARAMS['lp_ymin']
    src_max = 10**config.NN_PARAMS['lp_ymax']
    target_min = config.NN_PARAMS['rescale_target_min']
    target_max = config.NN_PARAMS['rescale_target_max']
    return target_min + (x - src_min) * (target_max - target_min) / (src_max - src_min)


# ==============================================================================
#                       MULTI-LAYER PERCEPTRON (MLP) CLASS
# ==============================================================================
# Neural network with 2 inputs, 1 hidden layer, and 1 output
# ==============================================================================

class mlp():
    def __init__(self, hidden, noutput):
        """
        Initialize MLP network
        
        Parameters:
        -----------
        hidden : list of str
            List of activation functions for hidden nodes (['HP', 'LP', ...])
        noutput : str
            Activation function for output node ('HP' or 'LP')
        """
        self.hidden = hidden
        self.noutput = noutput
        self.nhidden = len(hidden)
        
        # Set up activation functions for hidden nodes
        self.nodes = []
        for n in hidden:
            if n == 'HP':
                self.nodes.append(response_hp_configured)
            else:
                self.nodes.append(response_lp_configured)
    
    def forward(self, I, wH, wO):
        """
        Forward pass through the network
        
        Parameters:
        -----------
        I : array
            Input vector [input1, input2]
        wH : array
            Hidden layer weights, shape (n_hidden, 2)
        wO : array
            Output layer weights, shape (n_hidden,)
        
        Returns:
        --------
        output : float
            Network output value
        """
        activationsI = np.zeros([2]) 
        activationsH = np.zeros([self.nhidden]) 
        activationsO = np.zeros([1]) 
        
        # Input layer - copy inputs
        activationsI[0] = I[0]
        activationsI[1] = I[1]
        
        # Hidden layer - apply activation functions and rescale
        for ii in range(self.nhidden):
            # Weighted sum of inputs
            weighted_sum = wH[ii,0]*activationsI[0] + wH[ii,1]*activationsI[1]
            # Apply activation function
            activationsH[ii] = self.nodes[ii](weighted_sum)
            # Rescale to target range
            if self.nodes[ii] is response_hp_configured:
                activationsH[ii] = rescale_node_HP(activationsH[ii])
            else:
                activationsH[ii] = rescale_node_LP(activationsH[ii])
        
        # Output layer - apply activation function
        if self.noutput == 'HP':
            activationsO[0] = response_hp_configured(np.dot(wO, activationsH))
        else:
            activationsO[0] = response_lp_configured(np.dot(wO, activationsH))
        
        return activationsO[0]


# ==============================================================================
#                       SPATIAL WEIGHT CALCULATION
# ==============================================================================
# Calculate connection weights based on spatial positions using diffusion model
# ==============================================================================

def calculate_weights(x_indices, y_indices, config):
    """
    Calculate all connection weights based on spatial positions
    
    Network layout: [input1, input2, hidden_nodes..., output]
    Indices:        [0,      1,      2 to n-2,        n-1]
    
    Weights calculated:
    - From input1 and input2 to each hidden node
    - From each hidden node to output
    
    Parameters:
    -----------
    x_indices : array
        X coordinates of all nodes
    y_indices : array
        Y coordinates of all nodes
    config : Config
        Configuration object with spatial parameters
        
    Returns:
    --------
    weights : array
        All connection weights flattened
    """
    # Get spatial parameters
    diff_co = config.SPATIAL_PARAMS['diff_coefficient']
    time = config.SPATIAL_PARAMS['time_step']
    spatial_scale = config.SPATIAL_PARAMS['spatial_scale']
    
    num_indices = len(x_indices)
    num_hidden = num_indices - 3  # Total nodes - 2 inputs - 1 output
    
    # Total weights = (2 inputs * num_hidden) + (num_hidden * 1 output)
    num_weights = 2 * num_hidden + num_hidden
    weights = np.zeros(num_weights)
    index_pairs = []
    
    # Step 1: Weights from inputs (indices 0, 1) to hidden nodes (indices 2 to n-2)
    for i in range(2, num_indices - 1):
        index_pairs.append((i, 0))  # hidden <- input1
        index_pairs.append((i, 1))  # hidden <- input2
    
    # Step 2: Weights from hidden nodes (indices 2 to n-2) to output (index n-1)
    last_index = num_indices - 1
    for i in range(2, last_index):
        index_pairs.append((last_index, i))  # output <- hidden
    
    # Calculate spatial distances
    delX = np.abs(np.array([x_indices[i] - x_indices[j] for i, j in index_pairs])) * spatial_scale
    delY = np.abs(np.array([y_indices[i] - y_indices[j] for i, j in index_pairs])) * spatial_scale
    
    # Calculate weights using complementary error function (diffusion model)
    # Weight = erfc(delX / sqrt(4*D*t)) * erfc(delY / sqrt(4*D*t))
    for ii in range(len(index_pairs)):
        test1 = math.erfc(delX[ii] / (2 * (diff_co * time) ** 0.5))
        test2 = math.erfc(delY[ii] / (2 * (diff_co * time) ** 0.5))
        weights[ii] = test1 * test2
    
    return weights


def check_minimum_distance(x_indices, y_indices, min_distance):
    """
    Check if all nodes are at least min_distance apart
    
    Parameters:
    -----------
    x_indices : array
        X coordinates of all nodes
    y_indices : array
        Y coordinates of all nodes
    min_distance : float
        Minimum required distance between any two nodes
        
    Returns:
    --------
    valid : bool
        True if all distances >= min_distance, False otherwise
    """
    for i in range(len(x_indices)):
        for j in range(i + 1, len(x_indices)):
            dist = np.sqrt((x_indices[i] - x_indices[j])**2 + 
                          (y_indices[i] - y_indices[j])**2)
            if dist < min_distance:
                return False
    return True


# ==============================================================================
#                       LOSS FUNCTION
# ==============================================================================
# Mean squared error in log space
# ==============================================================================

def loss_fn(X, Y, w, network, ndata, num_hidden_nodes):
    """
    Calculate loss (mean squared error in log space)
    
    Parameters:
    -----------
    X : array
        Input data, shape (ndata, 2)
    Y : array
        Target outputs, shape (ndata,)
    w : array
        Connection weights (flattened)
    network : mlp
        Neural network object
    ndata : int
        Number of data points
    num_hidden_nodes : int
        Number of hidden nodes
        
    Returns:
    --------
    loss : float
        Mean squared error in log space
    """
    Yhat = np.zeros([ndata])
    w = np.array(w)
    
    # Reshape weights into matrices
    wH = w[0:2*num_hidden_nodes].reshape(num_hidden_nodes, 2)  # Hidden weights
    wO = w[2*num_hidden_nodes:]                                 # Output weights
    
    # Generate predictions
    for ii in range(ndata):
        Yhat[ii] = network.forward(X[ii,:], wH, wO)
    
    # Calculate loss in log space
    return ((np.log(1 + Yhat) - np.log(1 + Y))**2).mean()


# ==============================================================================
#                       GENETIC ALGORITHM
# ==============================================================================
# Optimize network topology and spatial positions simultaneously
# ==============================================================================

def run_genetic_algorithm(X, Y, ndata, config, gate_name, verbose=True):
    """
    Run genetic algorithm to optimize neural network for logic gate
    
    Optimizes both:
    1. Network topology (number and type of hidden nodes)
    2. Spatial positions of all nodes
    
    Parameters:
    -----------
    X : array
        Input data, shape (ndata, 2)
    Y : array
        Target outputs, shape (ndata,)
    ndata : int
        Number of training samples
    config : Config
        Configuration object
    gate_name : str
        Name of logic gate being optimized
    verbose : bool
        Print progress updates
        
    Returns:
    --------
    log_w : array
        Optimized connection weights
    best_network_topology : list
        Best network topology found
    best_x_locations : array
        Optimized X positions
    best_y_locations : array
        Optimized Y positions
    """
    
    # --------------------------------------------------------------------------
    # Extract parameters from config
    # --------------------------------------------------------------------------
    ga_params = config.GA_PARAMS
    spatial_params = config.SPATIAL_PARAMS
    
    ngen = ga_params['ngen']
    npop = ga_params['npop']
    pop_cut = int(npop * ga_params['pop_cut_ratio'])
    mutation_prob = ga_params['mutation_prob']
    min_network_size = ga_params['min_network_size']
    max_network_size = ga_params['max_network_size']
    
    segment_mutation = int(npop * ga_params['segment_ratio'])
    segment_pos_mutation = int(npop * ga_params['segment_pos_ratio'])
    nrecomb = int(npop * ga_params['recombination_ratio'])
    
    min_distance = spatial_params['min_distance']
    input_x_range = spatial_params['input_nodes_x_range']
    input_y_range = spatial_params['input_nodes_y_range']
    hidden_x_range = spatial_params['hidden_nodes_x_range']
    hidden_y_range = spatial_params['hidden_nodes_y_range']
    output_x = spatial_params['output_node_x_range'][0]
    output_y = spatial_params['output_node_y_range'][0]
    new_x_range = spatial_params['new_node_x_range']
    new_y_range = spatial_params['new_node_y_range']
    
    act_func = config.NN_PARAMS['activation_functions']
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Running GA for {gate_name}")
        print(f"{'='*70}")
        print(f"Generations: {ngen}, Population: {npop}, Training samples: {ndata}")
    
    # --------------------------------------------------------------------------
    # Initialize population arrays
    # --------------------------------------------------------------------------
    network_topology = [[None for _ in range(npop)] for _ in range(ngen)]
    x_indices_pop = [[None for _ in range(npop)] for _ in range(ngen)]
    y_indices_pop = [[None for _ in range(npop)] for _ in range(ngen)]
    weights = [[None for _ in range(npop)] for _ in range(ngen)]
    fitness = np.zeros([ngen, npop])
    
    # --------------------------------------------------------------------------
    # Generation 0: Random initialization
    # --------------------------------------------------------------------------
    if verbose:
        print("\nInitializing population...")
    
    for jj in range(npop):
        # Random network topology
        num_hidden_nodes = np.random.randint(min_network_size, max_network_size + 1)
        hidden_nodes = list(np.random.choice(act_func, num_hidden_nodes))
        output_node = np.random.choice(act_func)
        network_topology[0][jj] = hidden_nodes + [output_node]
        
        # Random spatial positions (with minimum distance constraint)
        attempts = 0
        max_attempts = 1000
        
        while attempts < max_attempts:
            x_indices = []
            y_indices = []
            
            # Input positions
            x_indices.append(np.random.randint(*input_x_range))
            y_indices.append(np.random.randint(*input_y_range))
            x_indices.append(np.random.randint(*input_x_range))
            y_indices.append(np.random.randint(*input_y_range))
            
            # Hidden node positions
            for _ in range(num_hidden_nodes):
                x_indices.append(np.random.randint(*hidden_x_range))
                y_indices.append(np.random.randint(*hidden_y_range))
            
            # Output position (fixed)
            x_indices.append(output_x)
            y_indices.append(output_y)
            
            # Check minimum distance constraint
            if check_minimum_distance(x_indices, y_indices, min_distance):
                break
            attempts += 1
        
        x_indices_pop[0][jj] = x_indices
        y_indices_pop[0][jj] = y_indices
        weights[0][jj] = calculate_weights(x_indices, y_indices, config)
        
        # Calculate fitness
        hidden_nodes = network_topology[0][jj][:-1]
        noutput = network_topology[0][jj][-1]
        network = mlp(hidden_nodes, noutput)
        num_hidden_nodes = len(hidden_nodes)
        fitness[0, jj] = loss_fn(X, Y, weights[0][jj], network, ndata, num_hidden_nodes)
    
    if verbose:
        print(f"Initial generation complete. Best fitness: {np.min(fitness[0, :]):.6f}")
    
    # --------------------------------------------------------------------------
    # Evolution loop
    # --------------------------------------------------------------------------
    for ii in range(1, ngen):
        print(f"\nGeneration {ii}/{ngen-1}:")
        
        # ----------------------------------------------------------------------
        # Selection: Keep top performers (elitism)
        # ----------------------------------------------------------------------
        srt = np.argsort(fitness[ii-1, :])
        for attr in [x_indices_pop, y_indices_pop, network_topology, weights]:
            attr[ii][0:pop_cut] = [attr[ii-1][index] for index in srt[0:pop_cut]]
            attr[ii][pop_cut:] = [attr[ii-1][index] for index in srt[0:pop_cut]] 
        
        fitness[ii, 0:pop_cut] = [fitness[ii-1, index] for index in srt[0:pop_cut]]
        
        # ----------------------------------------------------------------------
        # Mutation: Topology (first segment)
        # ----------------------------------------------------------------------
        for jj in range(pop_cut, pop_cut + segment_mutation):
            if np.random.rand() < mutation_prob:
                position = np.random.randint(0, len(network_topology[ii][jj]) - 1)
                operation = np.random.choice(['add', 'delete', 'modify'])
                
                if operation == 'add' and len(network_topology[ii][jj]) < max_network_size:
                    node_to_add = np.random.choice(act_func)
                    network_topology[ii][jj] = np.insert(network_topology[ii][jj], position, node_to_add).tolist()                    
                elif operation == 'delete' and len(network_topology[ii][jj]) > min_network_size:
                    network_topology[ii][jj] = np.delete(network_topology[ii][jj], position).tolist()
                elif operation == 'modify':
                     if network_topology[ii][jj][position] == 'LP':
                         network_topology[ii][jj][position] = 'HP'
                     elif network_topology[ii][jj][position] == 'HP':
                         network_topology[ii][jj][position] = 'LP'
        
        # ----------------------------------------------------------------------
        # Mutation: Positions (second segment)
        # ----------------------------------------------------------------------
        for jj in range(pop_cut + segment_mutation, pop_cut + segment_mutation + segment_pos_mutation):
            if np.random.uniform(0, 1) < mutation_prob:
                hidden_nodes = network_topology[ii][jj][:-1]
                num_hidden_nodes = len(hidden_nodes)
                
                # Perturb X position
                ipertx = np.random.randint(0, num_hidden_nodes + 2)
                new_x_indices = np.random.normal(loc=x_indices_pop[ii][jj][ipertx], scale=2, size=1).astype(int)
                x_indices_pop[ii][jj][ipertx] = new_x_indices
                
                # Perturb Y position
                iperty = np.random.randint(0, num_hidden_nodes + 2)
                new_y_indices = np.random.normal(loc=y_indices_pop[ii][jj][iperty], scale=2, size=1).astype(int)
                y_indices_pop[ii][jj][iperty] = new_y_indices
        
        # ----------------------------------------------------------------------
        # Replace worst performers with best
        # ----------------------------------------------------------------------
        for jj in range(npop - segment_mutation, npop):
            index = jj - (npop - segment_mutation)
            network_topology[ii][jj] = network_topology[ii][index]
            x_indices_pop[ii][jj] = x_indices_pop[ii][index]
            y_indices_pop[ii][jj] = y_indices_pop[ii][index]
        
        # ----------------------------------------------------------------------
        # Recombination
        # ----------------------------------------------------------------------
        irecomb1 = np.random.choice(range(pop_cut, npop), size=nrecomb, replace=True)
        irecomb2 = np.random.choice(range(pop_cut, npop), size=nrecomb, replace=True)
        for jj in range(nrecomb):
            ntwrk1 = network_topology[ii][irecomb1[jj]][:]
            ntwrk2 = network_topology[ii][irecomb2[jj]][:]
            if len(ntwrk1) + len(ntwrk2) >= min_network_size:
                new_recomb_network = np.concatenate((ntwrk1[0:2], ntwrk2[2:]))
            else:
                new_recomb_network = np.concatenate((ntwrk1[0:1], ntwrk2[1:]))
            network_topology[ii][irecomb1[jj]] = new_recomb_network
        
        # ----------------------------------------------------------------------
        # Adjust position arrays if topology changed
        # ----------------------------------------------------------------------
        for jj in range(pop_cut,npop):
            hidden_nodes = network_topology[ii][jj][:-1]
            num_hidden_nodes = len(hidden_nodes)
            expected_length = num_hidden_nodes + 3  # 2 inputs + hidden + 1 output
            
            if len(x_indices_pop[ii][jj]) != expected_length:
                n_add_delete = expected_length - len(x_indices_pop[ii][jj])
                
                if n_add_delete > 0:  # Add nodes
                    for _ in range(n_add_delete):
                        new_x_index = np.random.randint(*new_x_range)
                        new_y_index = np.random.randint(*new_y_range)
                        # Insert before output (last position)
                        x_indices_pop[ii][jj]=np.insert(x_indices_pop[ii][jj], -1, new_x_index)
                        y_indices_pop[ii][jj]=np.insert(y_indices_pop[ii][jj], -1, new_y_index)
                else:  # Delete nodes
                    n_delete = abs(n_add_delete)
                    # Delete from hidden nodes only (indices 2 to -2)
                    valid_indices = list(range(2, len(x_indices_pop[ii][jj]) - 1))
                    delete_indices = np.random.choice(valid_indices, n_delete, replace=False)
                    x_indices_pop[ii][jj] = np.delete(x_indices_pop[ii][jj], delete_indices)
                    y_indices_pop[ii][jj] = np.delete(y_indices_pop[ii][jj], delete_indices)
        
        # ----------------------------------------------------------------------
        # Recalculate fitness for modified individuals
        # ----------------------------------------------------------------------
        for jj in range(pop_cut, npop):
            hidden_nodes = network_topology[ii][jj][:-1]
            noutput = network_topology[ii][jj][-1]
            num_hidden_nodes = len(hidden_nodes)
            network = mlp(hidden_nodes, noutput)
            weights[ii][jj] = calculate_weights(x_indices_pop[ii][jj], y_indices_pop[ii][jj], config)
            fitness[ii, jj] = loss_fn(X, Y, weights[ii][jj], network, ndata, num_hidden_nodes)
        
        print(f"  Best fitness: {np.min(fitness[ii, :]):.6f}")
    
    # --------------------------------------------------------------------------
    # Extract top N solutions
    # --------------------------------------------------------------------------
    final_gen = ngen - 1
    srt = np.argsort(fitness[final_gen, :])
    
    # Number of top solutions to save
    top_n = min(config.OUTPUT_PARAMS['save_top_n'], npop)
    
    # Extract best solution (for backward compatibility)
    log_w = weights[final_gen][srt[0]][:]
    best_network_topology = network_topology[final_gen][srt[0]][:]
    best_x_locations = x_indices_pop[final_gen][srt[0]][:]
    best_y_locations = y_indices_pop[final_gen][srt[0]][:]
    
    # Extract top N solutions
    top_solutions = []
    for i in range(top_n):
        idx = srt[i]
        solution = {
            'rank': i + 1,
            'fitness': fitness[final_gen, idx],
            'weights': weights[final_gen][idx][:],
            'topology': network_topology[final_gen][idx][:],
            'x_positions': x_indices_pop[final_gen][idx][:],
            'y_positions': y_indices_pop[final_gen][idx][:]
        }
        top_solutions.append(solution)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Optimization complete!")
        print(f"Final best fitness: {fitness[final_gen, srt[0]]:.6f}")
        print(f"Best topology: {best_network_topology}")
        print(f"Top {top_n} solutions extracted")
        print(f"{'='*70}")
    
    return log_w, best_network_topology, best_x_locations, best_y_locations, top_solutions


# ==============================================================================
#                       EVALUATION FUNCTION
# ==============================================================================
# Test trained network on standard logic gate inputs
# ==============================================================================

def evaluate_logic_gate(network, weights, gate_name="Unknown"):
    """
    Evaluate trained network on standard logic inputs
    
    Parameters:
    -----------
    network : mlp
        Trained neural network
    weights : array
        Optimized connection weights
    gate_name : str
        Name of logic gate
        
    Returns:
    --------
    results : list
        Output values for each test input
    """
    test_inputs = config.ANALYSIS_PARAMS['test_inputs']
    
    print(f"\nEvaluating {gate_name} Gate:")
    print("Input\t\tOutput\t\tLog10(Output)")
    print("-" * 45)
    
    wH = weights[0:2*network.nhidden].reshape(network.nhidden, 2)
    wO = weights[2*network.nhidden:]
    
    results = []
    for inp in test_inputs:
        output = network.forward(inp, wH, wO)
        log_output = np.log10(output) if output > 0 else float('-inf')
        results.append(output)
        print(f"{inp}\t{output:.2e}\t{log_output:.2f}")
    
    return results

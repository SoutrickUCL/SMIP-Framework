# ==============================================================================
# SPATIAL_LOGIC.PY - Core Algorithm for 4-Input Binary Classifier
# ==============================================================================

import numpy as np
import math
from config import Config

config = Config()

# ==============================================================================
#                       ACTIVATION FUNCTIONS
# ==============================================================================

def response_hp(x, ymin, ymax, K, n):
    """High-pass response function"""
    return 10**( ymin + (ymax-ymin)*( (x**n)/( K**n + x**n) ) )

def response_lp(x, ymin, ymax, K, n):
    """Low-pass response function"""
    return 10**( ymin + (ymax-ymin)*( K**n /( K**n + x**n) ) )

def response_hp_configured(x):
    """HP with config parameters"""
    return response_hp(x, config.NN_PARAMS['hp_ymin'], config.NN_PARAMS['hp_ymax'],
                      config.NN_PARAMS['hp_K'], config.NN_PARAMS['hp_n'])

def response_lp_configured(x):
    """LP with config parameters"""
    return response_lp(x, config.NN_PARAMS['lp_ymin'], config.NN_PARAMS['lp_ymax'],
                      config.NN_PARAMS['lp_K'], config.NN_PARAMS['lp_n'])

# ==============================================================================
#                       RESCALING FUNCTIONS
# ==============================================================================

def rescale_node_HP(x):
    """Rescale HP output to target range"""
    src_min = 10**config.NN_PARAMS['hp_ymin']
    src_max = 10**config.NN_PARAMS['hp_ymax']
    target_min = config.NN_PARAMS['rescale_target_min']
    target_max = config.NN_PARAMS['rescale_target_max']
    return target_min + (x - src_min) * (target_max - target_min) / (src_max - src_min)

def rescale_node_LP(x):
    """Rescale LP output to target range"""
    src_min = 10**config.NN_PARAMS['lp_ymin']
    src_max = 10**config.NN_PARAMS['lp_ymax']
    target_min = config.NN_PARAMS['rescale_target_min']
    target_max = config.NN_PARAMS['rescale_target_max']
    return target_min + (x - src_min) * (target_max - target_min) / (src_max - src_min)

# ==============================================================================
#                       MLP NETWORK CLASS (4 INPUTS)
# ==============================================================================

class mlp():
    """Multi-layer perceptron with 4 inputs, 1 hidden layer, 1 output"""
    
    def __init__(self, hidden, output_node):
        self.hidden = hidden
        self.output_node = output_node
        self.nhidden = len(hidden)
        
        # Setup activation functions
        self.nodes = []
        for node_type in hidden:
            if node_type == 'HP':
                self.nodes.append(response_hp_configured)
            else:
                self.nodes.append(response_lp_configured)
    
    def forward(self, I, wH, wO):
        """
        Forward pass through network
        I: 4 inputs
        wH: hidden weights (nhidden x 4)
        wO: output weights (nhidden x 1)
        """
        # Input layer (4 inputs)
        activationsI = np.zeros(4)
        activationsI[0] = I[0]
        activationsI[1] = I[1]
        activationsI[2] = I[2]
        activationsI[3] = I[3]
        
        # Hidden layer
        activationsH = np.zeros(self.nhidden)
        for ii in range(self.nhidden):
            # Sum weighted inputs
            hidden_input = sum(wH[ii, jj] * activationsI[jj] for jj in range(4))
            # Apply activation
            activationsH[ii] = self.nodes[ii](hidden_input)
            # Rescale
            if self.nodes[ii] is response_hp_configured:
                activationsH[ii] = rescale_node_HP(activationsH[ii])
            elif self.nodes[ii] is response_lp_configured:
                activationsH[ii] = rescale_node_LP(activationsH[ii])
        
        # Output layer
        if self.output_node == 'HP':
            output = response_hp_configured(np.dot(wO, activationsH))
        else:
            output = response_lp_configured(np.dot(wO, activationsH))
        
        return output

# ==============================================================================
#                       SPATIAL WEIGHT CALCULATION (4 INPUTS)
# ==============================================================================

def calculate_weights_from_positions(x_indices, y_indices, num_hidden_nodes):
    """
    Calculate weights from spatial positions for 4-input network
    
    Layout: [input1, input2, input3, input4, hidden_nodes..., output]
    
    Returns weights for:
    - Input to Hidden: 4 * num_hidden_nodes weights
    - Hidden to Output: num_hidden_nodes weights
    """
    diff_co = config.SPATIAL_PARAMS['diff_coefficient']
    time = config.SPATIAL_PARAMS['time_step']
    spatial_scale = config.SPATIAL_PARAMS['spatial_scale']
    
    # Index structure
    input_indices = [0, 1, 2, 3]
    hidden_start = 4
    hidden_end = 4 + num_hidden_nodes
    output_idx = len(x_indices) - 1
    
    # Build index pairs
    index_pairs = []
    # Input to hidden connections
    for h_idx in range(hidden_start, hidden_end):
        for inp_idx in input_indices:
            index_pairs.append((h_idx, inp_idx))
    # Hidden to output connections
    for h_idx in range(hidden_start, hidden_end):
        index_pairs.append((output_idx, h_idx))
    
    # Calculate distances
    total_weights = len(index_pairs)
    weights = np.zeros(total_weights)
    
    delX = np.abs(np.array([x_indices[i] - x_indices[j] for i, j in index_pairs])) * spatial_scale
    delY = np.abs(np.array([y_indices[i] - y_indices[j] for i, j in index_pairs])) * spatial_scale
    
    # Calculate weights using diffusion model
    for ii in range(total_weights):
        t1 = math.erfc(delX[ii] / (2 * (diff_co * time) ** 0.5))
        t2 = math.erfc(delY[ii] / (2 * (diff_co * time) ** 0.5))
        weights[ii] = t1 * t2
    
    return weights

def calculate_distance(x_indices, y_indices, min_distance):
    """Check if all nodes are at least min_distance apart"""
    for i in range(len(x_indices)):
        for j in range(i + 1, len(x_indices)):
            dist = ((x_indices[i] - x_indices[j]) ** 2 + 
                   (y_indices[i] - y_indices[j]) ** 2) ** 0.5
            if dist < min_distance:
                return False
    return True

# ==============================================================================
#                       LOSS FUNCTION (4 INPUTS)
# ==============================================================================

def loss_fn(X, Y, w, network, ndata, num_hidden_nodes):
    """
    Calculate RMSLE loss
    X: (ndata, 4) - 4 inputs per sample
    """
    Yhat = np.zeros_like(Y)
    w = np.array(w)
    
    # Reshape weights for 4 inputs
    wH = w[0:4*num_hidden_nodes].reshape(num_hidden_nodes, 4)
    wO = w[4*num_hidden_nodes:]
    
    for ii in range(ndata):
        Yhat[ii] = network.forward(X[ii,:], wH, wO)
    
    return ((np.log(1+Yhat) - np.log(1+Y))**2).mean()

# ==============================================================================
#                       GENETIC ALGORITHM WITH TOP N SOLUTIONS
# ==============================================================================

def run_genetic_algorithm(X, Y, ndata, config, operation_name, verbose=True):
    """
    Run genetic algorithm to optimize network topology and spatial positions
    Returns: best_weights, best_topology, best_x_pos, best_y_pos, top_solutions
    """
    
    # Extract parameters from config
    ngen = config.GA_PARAMS['ngen']
    npop = config.GA_PARAMS['npop']
    mutation_prob = config.GA_PARAMS['mutation_prob']
    pop_cut = int(npop * config.GA_PARAMS['pop_cut_ratio'])
    min_net_size = config.GA_PARAMS['min_network_size']
    max_net_size = config.GA_PARAMS['max_network_size']
    min_distance = config.SPATIAL_PARAMS['min_distance']
    
    # Position ranges
    input_x_range = config.SPATIAL_PARAMS['input_nodes_x_range']
    input_y_range = config.SPATIAL_PARAMS['input_nodes_y_range']
    hidden_x_range = config.SPATIAL_PARAMS['hidden_nodes_x_range']
    hidden_y_range = config.SPATIAL_PARAMS['hidden_nodes_y_range']
    output_x = config.SPATIAL_PARAMS['output_node_x_range'][0]
    output_y = config.SPATIAL_PARAMS['output_node_y_range'][0]
    new_node_x_range = config.SPATIAL_PARAMS['new_node_x_range']
    new_node_y_range = config.SPATIAL_PARAMS['new_node_y_range']
    
    if verbose:
        print(f"\nStarting optimization for {operation_name}")
        print(f"Generations: {ngen}, Population: {npop}")
    
    # Initialize storage
    fitness = np.zeros([ngen, npop])
    network_topology = [[None for _ in range(npop)] for _ in range(ngen)]
    weights = [[None for _ in range(npop)] for _ in range(ngen)]
    x_indices_pop = [[None for _ in range(npop)] for _ in range(ngen)]
    y_indices_pop = [[None for _ in range(npop)] for _ in range(ngen)]
    
    act_func = config.NN_PARAMS['activation_functions']
    
    # Initialize generation 0
    network_topology[0] = [[str(np.random.choice(act_func)) for _ in range(min_net_size)] 
                          for _ in range(npop)]
    
    # Initialize positions for generation 0
    for pop in range(npop):
        num_hidden = len(network_topology[0][pop])
        
        while True:
            # 4 inputs + hidden nodes + 1 output
            x_pos = (list(np.random.randint(*input_x_range, 4)) +
                    list(np.random.randint(*hidden_x_range, num_hidden)) +
                    [output_x])
            y_pos = (list(np.random.randint(*input_y_range, 4)) +
                    list(np.random.randint(*hidden_y_range, num_hidden)) +
                    [output_y])
            
            if calculate_distance(x_pos, y_pos, min_distance):
                break
        
        x_indices_pop[0][pop] = x_pos
        y_indices_pop[0][pop] = y_pos
        
        # Calculate weights
        weights[0][pop] = calculate_weights_from_positions(x_pos, y_pos, num_hidden)
    
    # Calculate initial fitness
    for pop in range(npop):
        topology = network_topology[0][pop]
        hidden_nodes = topology[:-1]
        output_node = topology[-1]
        network = mlp(hidden_nodes, output_node)
        fitness[0, pop] = loss_fn(X, Y, weights[0][pop], network, ndata, len(hidden_nodes))
    
    if verbose:
        print(f"Generation 0: Best fitness = {np.min(fitness[0, :]):.6f}")
    
    # Evolution loop
    for gen in range(1, ngen):
        # Sort by fitness
        srt = np.argsort(fitness[gen-1, :])
        
        # Elitism - keep top performers
        for pop in range(pop_cut):
            network_topology[gen][pop] = network_topology[gen-1][srt[pop]].copy()
            weights[gen][pop] = weights[gen-1][srt[pop]].copy()
            x_indices_pop[gen][pop] = x_indices_pop[gen-1][srt[pop]].copy()
            y_indices_pop[gen][pop] = y_indices_pop[gen-1][srt[pop]].copy()
            fitness[gen, pop] = fitness[gen-1, srt[pop]]
        
        # Generate new population
        for pop in range(pop_cut, npop):
            parent_idx = np.random.randint(0, pop_cut)
            
            # Copy from parent
            new_topology = network_topology[gen][parent_idx].copy()
            new_x_pos = x_indices_pop[gen][parent_idx].copy()
            new_y_pos = y_indices_pop[gen][parent_idx].copy()
            
            # Mutation
            if np.random.rand() < mutation_prob:
                mutation_type = np.random.choice(['add', 'delete', 'modify'])
                
                if mutation_type == 'add' and len(new_topology) - 1 < max_net_size:
                    # Add node
                    insert_pos = np.random.randint(0, len(new_topology))
                    new_topology.insert(insert_pos, np.random.choice(act_func))
                    
                    new_x = np.random.randint(*new_node_x_range)
                    new_y = np.random.randint(*new_node_y_range)
                    new_x_pos.insert(4 + insert_pos, new_x)
                    new_y_pos.insert(4 + insert_pos, new_y)
                
                elif mutation_type == 'delete' and len(new_topology) - 1 > min_net_size:
                    # Delete node
                    del_idx = np.random.randint(0, len(new_topology) - 1)
                    del new_topology[del_idx]
                    del new_x_pos[4 + del_idx]
                    del new_y_pos[4 + del_idx]
                
                elif mutation_type == 'modify':
                    # Modify activation function
                    mod_idx = np.random.randint(0, len(new_topology))
                    new_topology[mod_idx] = np.random.choice(act_func)
            
            # Position mutation
            segment_size = int(len(new_x_pos) * config.GA_PARAMS['segment_pos_ratio'])
            if segment_size > 0:
                start_idx = np.random.randint(0, len(new_x_pos) - segment_size + 1)
                for idx in range(start_idx, start_idx + segment_size):
                    if idx < 4:  # Input nodes
                        new_x_pos[idx] += np.random.randint(-500, 501)
                        new_y_pos[idx] += np.random.randint(-500, 501)
                    elif idx < len(new_x_pos) - 1:  # Hidden nodes
                        new_x_pos[idx] += np.random.randint(-500, 501)
                        new_y_pos[idx] += np.random.randint(-500, 501)
            
            # Check distance constraint
            if not calculate_distance(new_x_pos, new_y_pos, min_distance):
                # Reject - use parent
                network_topology[gen][pop] = network_topology[gen][parent_idx].copy()
                x_indices_pop[gen][pop] = x_indices_pop[gen][parent_idx].copy()
                y_indices_pop[gen][pop] = y_indices_pop[gen][parent_idx].copy()
                weights[gen][pop] = weights[gen][parent_idx].copy()
                fitness[gen, pop] = fitness[gen, parent_idx]
            else:
                # Accept mutation
                network_topology[gen][pop] = new_topology
                x_indices_pop[gen][pop] = new_x_pos
                y_indices_pop[gen][pop] = new_y_pos
                
                num_hidden = len(new_topology) - 1
                weights[gen][pop] = calculate_weights_from_positions(new_x_pos, new_y_pos, num_hidden)
                
                # Evaluate fitness
                hidden_nodes = new_topology[:-1]
                output_node = new_topology[-1]
                network = mlp(hidden_nodes, output_node)
                fitness[gen, pop] = loss_fn(X, Y, weights[gen][pop], network, ndata, num_hidden)
        
        if verbose and gen % 10 == 0:
            print(f"Generation {gen}: Best fitness = {np.min(fitness[gen, :]):.6f}")
    
    # Extract top N solutions
    final_gen = ngen - 1
    srt = np.argsort(fitness[final_gen, :])
    top_n = min(config.OUTPUT_PARAMS['save_top_n'], npop)
    
    # Best solution
    log_w = weights[final_gen][srt[0]][:]
    best_topology = network_topology[final_gen][srt[0]][:]
    best_x_pos = x_indices_pop[final_gen][srt[0]][:]
    best_y_pos = y_indices_pop[final_gen][srt[0]][:]
    
    # Top N solutions
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
        print(f"\nOptimization complete!")
        print(f"Final best fitness: {fitness[final_gen, srt[0]]:.6f}")
        print(f"Best topology: {best_topology}")
        print(f"Top {top_n} solutions extracted")
    
    return log_w, best_topology, best_x_pos, best_y_pos, top_solutions

# ==============================================================================
#                       EVALUATION FUNCTION
# ==============================================================================

def evaluate_classifier(network, weights, operation_name):
    """Evaluate classifier on all 10 numbers (0-9)"""
    from data_generator import create_binary_input_matrix, get_operation_function
    
    # Create binary inputs for 0-9
    on_cutoff = config.NN_PARAMS['on_cutoff']
    off_cutoff = config.NN_PARAMS['off_cutoff']
    X, X2 = create_binary_input_matrix(10, on_cutoff, off_cutoff)
    
    # Get expected outputs
    operation_func = get_operation_function(operation_name)
    Y_expected = operation_func(X2, on_cutoff, off_cutoff)
    
    # Calculate predictions
    num_hidden = len(network.hidden)
    weights = np.array(weights)
    wH = weights[0:4*num_hidden].reshape(num_hidden, 4)
    wO = weights[4*num_hidden:]
    
    Y_pred = np.zeros(10)
    for i in range(10):
        Y_pred[i] = network.forward(X[i,:], wH, wO)
    
    return Y_pred

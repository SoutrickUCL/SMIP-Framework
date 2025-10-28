# ==============================================================================
# SPATIAL_LOGIC.PY - Disease Classifier with 2 Hidden Layers
# ==============================================================================
# NOW PROPERLY USES CONFIG.PY AND SAVES TOP N SOLUTIONS!
# ==============================================================================

import numpy as np
from matplotlib import pyplot as plt
import math
from config import Config

# Load configuration
config = Config()

#######################################################################
#              Spatial Weight Calculation Functions
#######################################################################

def calculate_weights_from_positions(x_indices, y_indices, nHN1, nHN2):
    """
    Calculate all weights based on node positions.
    Layout: [input1, input2, input3, hidden1_nodes (nHN1), hidden2_nodes (nHN2), output]
    
    Weights needed:
    - Input to Hidden1: 3 * nHN1 weights
    - Hidden1 to Hidden2: nHN1 * nHN2 weights  
    - Hidden2 to Output: nHN2 weights
    """
    weights = []

    # Get spatial parameters from config
    diff_co = config.SPATIAL_PARAMS['diff_coefficient']
    time = config.SPATIAL_PARAMS['time_step']
    spatial_scale = config.SPATIAL_PARAMS['spatial_scale']
    
    # Indices structure: [0,1,2] = inputs, [3:3+nHN1] = hidden1, [3+nHN1:3+nHN1+nHN2] = hidden2, [-1] = output
    input_indices = [0, 1, 2]
    hidden1_start = 3
    hidden1_end = 3 + nHN1
    hidden2_start = hidden1_end
    hidden2_end = hidden2_start + nHN2
    output_idx = len(x_indices) - 1
    
    # 1. Weights from inputs to hidden1 (3 * nHN1 weights)
    for h1_idx in range(hidden1_start, hidden1_end):
        for inp_idx in input_indices:
            delX = abs(x_indices[h1_idx] - x_indices[inp_idx]) * spatial_scale
            delY = abs(y_indices[h1_idx] - y_indices[inp_idx]) * spatial_scale
            #print("delX:", delX / (2 * (diff_co * time) ** 0.5), "delY:", delY / (2 * (diff_co * time) ** 0.5))
            weight = math.erfc(delX / (2 * (diff_co * time) ** 0.5)) * \
                     math.erfc(delY / (2 * (diff_co * time) ** 0.5))
            weights.append(weight)
            #print("Weight (Input to Hidden1):", weight)
    
    # 2. Weights from hidden1 to hidden2 (nHN1 * nHN2 weights)
    for h2_idx in range(hidden2_start, hidden2_end):
        for h1_idx in range(hidden1_start, hidden1_end):
            delX = abs(x_indices[h2_idx] - x_indices[h1_idx]) * spatial_scale
            delY = abs(y_indices[h2_idx] - y_indices[h1_idx]) * spatial_scale
            #print("delX:", delX / (2 * (diff_co * time) ** 0.5), "delY:", delY / (2 * (diff_co * time) ** 0.5))
            weight = math.erfc(delX / (2 * (diff_co * time) ** 0.5)) * \
                     math.erfc(delY / (2 * (diff_co * time) ** 0.5))
            weights.append(weight)
            #print("Weight (Hidden1 to Hidden2):", weight)
    
    # 3. Weights from hidden2 to output (nHN2 weights)
    for h2_idx in range(hidden2_start, hidden2_end):
        delX = abs(x_indices[output_idx] - x_indices[h2_idx]) * spatial_scale
        delY = abs(y_indices[output_idx] - y_indices[h2_idx]) * spatial_scale
        #print("delX:", delX / (2 * (diff_co * time) ** 0.5), "delY:", delY / (2 * (diff_co * time) ** 0.5))
        weight = math.erfc(delX / (2 * (diff_co * time) ** 0.5)) * \
                 math.erfc(delY / (2 * (diff_co * time) ** 0.5))
        weights.append(weight)
        #print("Weight (Hidden2 to Output):", weight)

    return np.array(weights)

def calculate_distance(x_indices, y_indices, min_distance):
    """Check if all nodes are at least min_distance apart"""
    distances = []
    
    for i in range(len(x_indices)):
        for j in range(i + 1, len(x_indices)):
            dist = ((x_indices[i] - x_indices[j]) ** 2 + 
                   (y_indices[i] - y_indices[j]) ** 2) ** 0.5
            distances.append(dist)
    
    return all(distance >= min_distance for distance in distances)

#######################################################################
#              MLP Network Class
#######################################################################

class mlp():
    def __init__(self, hidden1, hidden2, noutput, act_func_params):
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.noutput = noutput
        self.nhidden1 = len(hidden1)
        self.nhidden2 = len(hidden2)
        self.act_func_params = list(act_func_params)

        self.nodes1 = [self.response_hp if n == 'HP' else self.response_lp for n in hidden1]
        self.nodes2 = [self.response_hp if n == 'HP' else self.response_lp for n in hidden2]

    def response_hp(self, x):
        KH, nH = self.act_func_params[0], self.act_func_params[1]
        x = np.clip(x, 0, None)
        return ( ( (x**nH)/( KH**nH + x**nH) ) )

    def response_lp(self, x):
        KL, nL = self.act_func_params[2], self.act_func_params[3]
        x = np.clip(x, 0, None)
        return (( KL**nL /( KL**nL + x**nL) ) )

    def forward(self, I, wH1, wH2, wO):
        Inputs = np.zeros(len(I)) 
        activationsH1 = np.zeros([self.nhidden1]) 
        activationsH2 = np.zeros([self.nhidden2]) 
        activationsO = np.zeros([1]) 

        Inputs[:] = I[:]

        for ii in range(self.nhidden1):
            activationsH1[ii] = self.nodes1[ii]( wH1[ii,0]*Inputs[0] + wH1[ii,1]*Inputs[1]+ wH1[ii,2]*Inputs[2] )
            activationsH1[ii] = rescale_output(activationsH1[ii])

        for ii in range(self.nhidden2):
            activationsH2[ii] = self.nodes2[ii](np.sum([wH2[ii, jj]*activationsH1[jj] for jj in range(len(activationsH1))]))
            activationsH2[ii] = rescale_output(activationsH2[ii])
            
        if self.noutput == 'HP':
          activationsO[0] = self.response_hp( np.dot(wO, activationsH2) )
        else:
          activationsO[0] = self.response_lp( np.dot(wO, activationsH2) )
        return (activationsO[0])

#######################################################################
#              Loss Function
#######################################################################

def loss_fn(X,Y,w,network, ndata, nHN1 , nHN2):
    Yhat = np.zeros_like(Y, dtype=float)
    w = np.array(w)
    
    wH1 = w[0:3*nHN1].reshape(nHN1,3) 
    wH2=  w[3*nHN1:3*nHN1+nHN1*nHN2].reshape(nHN2,nHN1)
    wO = w[3*nHN1+nHN1*nHN2:]
    
    for ii in range(ndata):
        Yhat[ii] = network.forward(X[ii,:], wH1, wH2, wO)
        
    eps = 1e-8
    Yhat_clipped = np.clip(Yhat, -1 + eps, None)
    Y_clipped = np.clip(Y, -1 + eps, None)
    
    return ((np.log(1 + Yhat_clipped) - np.log(1 + Y_clipped))**2).mean()

#######################################################################
#              Genetic Algorithm with Spatial Positions
#######################################################################

def run_genetic_algorithm(X, Y, ndata):
    """
    Run genetic algorithm with 2 hidden layers
    NOW USES CONFIG AND RETURNS TOP N SOLUTIONS!
    """
    # Get parameters from config
    ngen = config.GA_PARAMS['ngen']
    npop = config.GA_PARAMS['npop']
    pop_cut = int(npop * config.GA_PARAMS['pop_cut_ratio'])
    nrecomb = int(npop * config.GA_PARAMS['recombination_ratio'])
    mutation_prob = config.GA_PARAMS['mutation_prob']
    min_distance = config.SPATIAL_PARAMS['min_distance']
    
    # Spatial parameters from config
    diff_co = config.SPATIAL_PARAMS['diff_coefficient']
    time = config.SPATIAL_PARAMS['time_step']
    spatial_scale = config.SPATIAL_PARAMS['spatial_scale']
    
    # Position ranges from config
    input_hidden_range = config.SPATIAL_PARAMS['input_nodes_x_range']
    output_x_fixed = config.SPATIAL_PARAMS['output_node_x_range'][0]
    output_y_fixed = config.SPATIAL_PARAMS['output_node_y_range'][0]
    
    # Network sizes from config
    nHN1_min = config.GA_PARAMS['min_hidden1_size']
    nHN1_max = config.GA_PARAMS['max_hidden1_size']
    nHN2_min = config.GA_PARAMS['min_hidden2_size']
    nHN2_max = config.GA_PARAMS['max_hidden2_size']
    
    print(f"\nStarting Disease Classifier GA")
    print(f"Generations: {ngen}, Population: {npop}")
    print(f"Hidden1 size: {nHN1_min}-{nHN1_max}, Hidden2 size: {nHN2_min}-{nHN2_max}")
    
    fitness = np.zeros([ngen, npop])
    act_func = config.NN_PARAMS['activation_functions']
    nHN1, nHN2, nO = nHN1_min, nHN2_min, 1
    
    # Network topology storage
    hidden1_topology = [[None for _ in range(npop)] for _ in range(ngen)]
    hidden1_topology[0] = [[str(np.random.choice(act_func)) for _ in range(nHN1)] for _ in range(npop)]

    hidden2_topology = [[None for _ in range(npop)] for _ in range(ngen)]
    hidden2_topology[0] = [[str(np.random.choice(act_func)) for _ in range(nHN2)] for _ in range(npop)]

    output_topology = [[None for _ in range(npop)] for _ in range(ngen)]
    output_topology[0] = [[str(np.random.choice(act_func)) for _ in range(nO)] for _ in range(npop)]
    
    # Spatial positions storage
    x_positions = [[None for _ in range(npop)] for _ in range(ngen)]
    y_positions = [[None for _ in range(npop)] for _ in range(ngen)]
    
    # Weights (calculated from positions)
    weights = [[None for _ in range(npop)] for _ in range(ngen)]
    
    # Activation function parameters
    act_func_params = [[None for _ in range(npop)] for _ in range(ngen)]
    for ii in range(npop):
        KH = 10 ** np.random.uniform(-7, -5)
        nH = np.random.randint(1, 10)
        KL = 10 ** np.random.uniform(-7, -5)
        nL = np.random.randint(1, 10)
        act_func_params[0][ii] = [KH, nH, KL, nL]
    
    #------------------------------------------------------------------------
    #           Initialize positions and calculate weights for first generation
    #------------------------------------------------------------------------
    print("Initializing first generation with spatial positions...")
    for jj in range(npop):
        if jj % 50000 == 0:
            print(f"  Initializing individual {jj}/{npop}")
        
        # Total nodes: 3 inputs + nHN1 hidden1 + nHN2 hidden2 + 1 output
        total_nodes = 3 + nHN1 + nHN2 + 1
        
        x_indices_pop_trail = np.zeros(total_nodes)
        y_indices_pop_trail = np.zeros(total_nodes)
        
        # Fix output node position
        x_indices_pop_trail[-1] = output_x_fixed
        y_indices_pop_trail[-1] = output_y_fixed
        
        attempts = 0
        max_attempts = 50000
        
        while attempts < max_attempts:
            # Input nodes (first 3)
            x_indices_pop_trail[0] = np.random.uniform(*input_hidden_range)  # X index of input 1
            x_indices_pop_trail[1] = np.random.uniform(*input_hidden_range)  # X index of input 2
            x_indices_pop_trail[2] = np.random.uniform(*input_hidden_range)  # X index of input 3
            
            y_indices_pop_trail[0] = np.random.uniform(*input_hidden_range)  # Y index of input 1
            y_indices_pop_trail[1] = np.random.uniform(*input_hidden_range)  # Y index of input 2
            y_indices_pop_trail[2] = np.random.uniform(*input_hidden_range)  # Y index of input 3
            
            # Hidden layer 1 nodes
            for i in range(3, 3 + nHN1):
                x_indices_pop_trail[i] = np.random.uniform(*input_hidden_range)
                y_indices_pop_trail[i] = np.random.uniform(*input_hidden_range)
            
            # Hidden layer 2 nodes
            for i in range(3 + nHN1, 3 + nHN1 + nHN2):
                x_indices_pop_trail[i] = np.random.uniform(*input_hidden_range)
                y_indices_pop_trail[i] = np.random.uniform(*input_hidden_range)
            
            # Check distance constraint
            if calculate_distance(x_indices_pop_trail, y_indices_pop_trail, min_distance):
                break
            
            attempts += 1
        
        if attempts == max_attempts:
            print(f"Warning: Individual {jj} reached max attempts for distance constraint")
        
        x_positions[0][jj] = x_indices_pop_trail.copy()
        y_positions[0][jj] = y_indices_pop_trail.copy()
        
        # Calculate weights from positions
        weights[0][jj] = calculate_weights_from_positions(
            x_indices_pop_trail, y_indices_pop_trail, nHN1, nHN2)
        
    
    #------------------------------------------------------------------------
    #           Calculate initial fitness
    #------------------------------------------------------------------------
    for jj in range(npop):
        hidden1 = hidden1_topology[0][jj][:]
        hidden2 = hidden2_topology[0][jj][:]
        output = output_topology[0][jj][:]
        network = mlp(hidden1, hidden2, output, act_func_params[0][jj])
        fitness[0,jj] = loss_fn(X, Y, weights[0][jj], network, ndata, nHN1, nHN2)

    #------------------------------------------------------------------------
    #              Generation Loop
    #------------------------------------------------------------------------
    for ii in range(1, ngen): 
        print(f"Generation: {ii}")
        srt = np.argsort(fitness[ii-1,:])
        
        # Selection - copy top performers
        x_positions[ii][0:pop_cut] = [x_positions[ii-1][index].copy() for index in srt[0:pop_cut]]
        x_positions[ii][pop_cut:] = [x_positions[ii-1][index].copy() for index in srt[0:pop_cut]]
        
        y_positions[ii][0:pop_cut] = [y_positions[ii-1][index].copy() for index in srt[0:pop_cut]]
        y_positions[ii][pop_cut:] = [y_positions[ii-1][index].copy() for index in srt[0:pop_cut]]
        
        weights[ii][0:pop_cut] = [weights[ii-1][index] for index in srt[0:pop_cut]]
        weights[ii][pop_cut:] = [weights[ii-1][index] for index in srt[0:pop_cut]]

        hidden1_topology[ii][0:pop_cut] = [hidden1_topology[ii-1][index] for index in srt[0:pop_cut]]
        hidden1_topology[ii][pop_cut:] = [hidden1_topology[ii-1][index] for index in srt[0:pop_cut]]

        hidden2_topology[ii][0:pop_cut] = [hidden2_topology[ii-1][index] for index in srt[0:pop_cut]]
        hidden2_topology[ii][pop_cut:] = [hidden2_topology[ii-1][index] for index in srt[0:pop_cut]]

        output_topology[ii][0:pop_cut] = [output_topology[ii-1][index] for index in srt[0:pop_cut]]
        output_topology[ii][pop_cut:] = [output_topology[ii-1][index] for index in srt[0:pop_cut]]

        act_func_params[ii][0:pop_cut] = [act_func_params[ii-1][index] for index in srt[0:pop_cut]]
        act_func_params[ii][pop_cut:] = [act_func_params[ii-1][index] for index in srt[0:pop_cut]]

        fitness[ii][0:pop_cut] = [fitness[ii-1][index] for index in srt[0:pop_cut]]

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #                               Mutation - Now mutating POSITIONS instead of weights
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        # Mutate topology for hidden layer 1
        for jj in range(pop_cut, npop):
            mutation_counter = np.random.uniform(0, 1)
            if mutation_counter < mutation_prob:
                position = np.random.randint(0, len(hidden1_topology[ii][jj])-1)
                operation = np.random.choice(['add', 'delete', 'modify'])
                if operation == 'add':
                    node_to_add = np.random.choice(['HP', 'LP'])
                    hidden1_topology[ii][jj] = np.insert(hidden1_topology[ii][jj], position, node_to_add).tolist()
                elif operation == 'delete':
                    if len(hidden1_topology[ii][jj]) > 2:
                        hidden1_topology[ii][jj] = np.delete(hidden1_topology[ii][jj], position).tolist()
                elif operation == 'modify':
                    if hidden1_topology[ii][jj][position] == 'LP':
                         hidden1_topology[ii][jj][position] = 'HP'
                    elif hidden1_topology[ii][jj][position] == 'HP':
                         hidden1_topology[ii][jj][position] = 'LP'
            else:
                # Mutate POSITION instead of weight (but NOT the output node!)
                nHN1 = len(hidden1_topology[ii][jj])
                nHN2 = len(hidden2_topology[ii][jj])
                total_nodes = 3 + nHN1 + nHN2 + 1
                
                # Randomly select a node to perturb position (excluding the output node at index -1)
                ipert = np.random.randint(0, total_nodes - 1)  # Exclude output node
                x_positions[ii][jj][ipert] = np.random.normal(loc=x_positions[ii][jj][ipert], scale=2)
                y_positions[ii][jj][ipert] = np.random.normal(loc=y_positions[ii][jj][ipert], scale=2)
                
                # Keep output position fixed
                x_positions[ii][jj][-1] = output_x_fixed
                y_positions[ii][jj][-1] = output_y_fixed

        # Mutate topology for hidden layer 2
        for jj in range(pop_cut, npop):
            mutation_counter = np.random.uniform(0, 1)
            if mutation_counter < mutation_prob:
                position = np.random.randint(0, len(hidden2_topology[ii][jj])-1)
                operation = np.random.choice(['add', 'delete', 'modify'])
                if operation == 'add':
                     node_to_add = np.random.choice(['HP', 'LP'])
                     hidden2_topology[ii][jj] = np.insert(hidden2_topology[ii][jj], position, node_to_add).tolist()
                elif operation == 'delete':
                     if len(hidden2_topology[ii][jj]) > 2:
                         hidden2_topology[ii][jj] = np.delete(hidden2_topology[ii][jj], position).tolist()
                elif operation == 'modify':
                     if hidden2_topology[ii][jj][position] == 'LP':
                         hidden2_topology[ii][jj][position] = 'HP'
                     elif hidden2_topology[ii][jj][position] == 'HP':
                         hidden2_topology[ii][jj][position] = 'LP'
            else:
                # Mutate POSITION (excluding output node)
                nHN1 = len(hidden1_topology[ii][jj])
                nHN2 = len(hidden2_topology[ii][jj])
                total_nodes = 3 + nHN1 + nHN2 + 1
                
                ipert = np.random.randint(0, total_nodes - 1)  # Exclude output node
                x_positions[ii][jj][ipert] = np.random.normal(loc=x_positions[ii][jj][ipert], scale=2)
                y_positions[ii][jj][ipert] = np.random.normal(loc=y_positions[ii][jj][ipert], scale=2)
                
                # Keep output position fixed
                x_positions[ii][jj][-1] = output_x_fixed
                y_positions[ii][jj][-1] = output_y_fixed

        # Mutate output topology
        for jj in range(pop_cut, npop):
            mutation_counter_output = np.random.uniform(0, 1)
            if mutation_counter_output < 0.05:
                if output_topology[ii][jj][:] == 'LP':
                    output_topology[ii][jj][:] = 'HP'
                elif output_topology[ii][jj][:] == 'HP':
                    output_topology[ii][jj][:] = 'LP'

        # Mutate activation function parameters
        for jj in range(pop_cut, npop):
            mutation_counter_output = np.random.uniform(0, 1)
            if mutation_counter_output < 0.05:
                ipert = np.random.randint(0, 3)
                new_act_function_param = np.random.normal(loc=list(act_func_params[ii][jj])[ipert], scale=0.1)
                act_func_params[ii][jj] = list(act_func_params[ii][jj])
                act_func_params[ii][jj][ipert] = new_act_function_param

        # Recombination for hidden layer 1
        irecomb1_H1 = np.random.choice(range(pop_cut, npop), size=nrecomb, replace=True)
        irecomb2_H1 = np.random.choice(range(pop_cut, npop), size=nrecomb, replace=True)
        for jj in range(nrecomb):
            ntwrk1 = hidden1_topology[ii][irecomb1_H1[jj]][:]
            ntwrk2 = hidden1_topology[ii][irecomb2_H1[jj]][:]
            if len(ntwrk1) + len(ntwrk2) >= 3:
                new_recomb_network1 = np.concatenate((ntwrk1[0:2], ntwrk2[2:]))
            else:
                new_recomb_network1 = np.concatenate((ntwrk1[0:1], ntwrk2[1:]))
            hidden1_topology[ii][irecomb1_H1[jj]] = new_recomb_network1.tolist()

        # Recombination for hidden layer 2
        irecomb1_H2 = np.random.choice(range(pop_cut, npop), size=nrecomb, replace=True)
        irecomb2_H2 = np.random.choice(range(pop_cut, npop), size=nrecomb, replace=True)
        for jj in range(nrecomb):
            ntwrk1 = hidden2_topology[ii][irecomb1_H2[jj]][:]
            ntwrk2 = hidden2_topology[ii][irecomb2_H2[jj]][:]
            if len(ntwrk1) + len(ntwrk2) >= 3:
                new_recomb_network2 = np.concatenate((ntwrk1[0:2], ntwrk2[2:]))
            else:
                new_recomb_network2 = np.concatenate((ntwrk1[0:1], ntwrk2[1:]))
            hidden2_topology[ii][irecomb1_H2[jj]] = new_recomb_network2.tolist()

        # Check and adjust positions array if topology changed
        for jj in range(pop_cut, npop):
            nHN1 = len(hidden1_topology[ii][jj])
            nHN2 = len(hidden2_topology[ii][jj])
            total_nodes_needed = 3 + nHN1 + nHN2 + 1
            
            current_nodes = len(x_positions[ii][jj])
            
            if current_nodes != total_nodes_needed:
                n_diff = total_nodes_needed - current_nodes
                
                if n_diff > 0:  # Need to add nodes
                    for _ in range(n_diff):
                        new_x = np.random.uniform(*input_hidden_range)
                        new_y = np.random.uniform(*input_hidden_range)
                        # Insert before the output node (at position -1)
                        x_positions[ii][jj] = np.insert(x_positions[ii][jj], -1, new_x)
                        y_positions[ii][jj] = np.insert(y_positions[ii][jj], -1, new_y)
                else:  # Need to delete nodes
                    n_delete = abs(n_diff)
                    # Valid indices: exclude inputs (0,1,2) and output (-1)
                    valid_indices = list(range(3, len(x_positions[ii][jj]) - 1))
                    delete_indices = np.random.choice(valid_indices, min(n_delete, len(valid_indices)), replace=False)
                    x_positions[ii][jj] = np.delete(x_positions[ii][jj], delete_indices)
                    y_positions[ii][jj] = np.delete(y_positions[ii][jj], delete_indices)
            
            # Ensure output position remains fixed after any modifications
            x_positions[ii][jj][-1] = output_x_fixed
            y_positions[ii][jj][-1] = output_y_fixed
            
            # Recalculate weights from positions
            weights[ii][jj] = calculate_weights_from_positions(
                x_positions[ii][jj], y_positions[ii][jj], nHN1, nHN2
            )

        # Recalculate fitness
        for jj in range(pop_cut, npop):
            nHN1 = len(hidden1_topology[ii][jj])
            nHN2 = len(hidden2_topology[ii][jj]) 
            network = mlp(hidden1_topology[ii][jj], hidden2_topology[ii][jj], 
                         output_topology[ii][jj], act_func_params[ii][jj]) 
            fitness[ii,jj] = loss_fn(X, Y, weights[ii][jj], network, ndata, nHN1, nHN2)
        
        print(f"\tBest Fitness: {np.min(fitness[ii,:])}")
        # Print best performing topology in the generation
        best_idx = np.argmin(fitness[ii,:])
        print(f"\tBest Hidden1 Topology: {hidden1_topology[ii][best_idx]}")
        print(f"\tBest Hidden2 Topology: {hidden2_topology[ii][best_idx]}")
        print(f"\tBest Output Topology: {output_topology[ii][best_idx]}")

    # Extract best solution
    final_gen = ngen - 1
    srt = np.argsort(fitness[final_gen,:])
    
    opt_weights = weights[final_gen][srt[0]][:]
    opt_hidden1_topology = hidden1_topology[final_gen][srt[0]][:]
    opt_hidden2_topology = hidden2_topology[final_gen][srt[0]][:]
    opt_output_topology = output_topology[final_gen][srt[0]][:]
    opt_act_func_params = list(act_func_params[final_gen][srt[0]])
    opt_x_positions = x_positions[final_gen][srt[0]][:]
    opt_y_positions = y_positions[final_gen][srt[0]][:]
    
    # Extract top N solutions
    top_n = min(config.OUTPUT_PARAMS['save_top_n'], npop)
    top_solutions = []
    for i in range(top_n):
        idx = srt[i]
        solution = {
            'rank': i + 1,
            'fitness': fitness[final_gen, idx],
            'weights': weights[final_gen][idx][:],
            'hidden1_topology': hidden1_topology[final_gen][idx][:],
            'hidden2_topology': hidden2_topology[final_gen][idx][:],
            'output_topology': output_topology[final_gen][idx][:],
            'act_func_params': list(act_func_params[final_gen][idx]),
            'x_positions': x_positions[final_gen][idx][:],
            'y_positions': y_positions[final_gen][idx][:]
        }
        top_solutions.append(solution)
    
    print(f"\nOptimization complete!")
    print(f"Final best fitness: {fitness[final_gen, srt[0]]:.6f}")
    print(f"Top {top_n} solutions extracted")
    
    return (opt_weights, opt_hidden1_topology, opt_hidden2_topology, 
            opt_output_topology, opt_act_func_params, opt_x_positions, opt_y_positions,
            top_solutions)

def rescale_output(arr, new_min=1e-8, new_max=1e-4):
    arr = np.array(arr)
    return arr * (new_max - new_min) + new_min

# ==============================================================================
# RUN.PY - Main Execution Script for 2-Input Logic Gates
# ==============================================================================
# This script handles data generation, optimization, testing, and visualization
# for 2-input logic gates (OR, AND, NOR, NAND, XOR).
# ==============================================================================

import os
import numpy as np
from matplotlib import pyplot as plt
import argparse
import sys
import spatial_logic as slg
import data_generator as datagen
from config import Config

# Clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# Load configuration
config = Config()

# ==============================================================================
#                       COMMAND LINE INTERFACE
# ==============================================================================

def print_usage():
    """Print usage information"""
    print("\n" + "="*70)
    print("LOGIC GATE NEURAL NETWORK OPTIMIZER (2 INPUTS)")
    print("="*70)
    print(f"Available gates: {', '.join(config.AVAILABLE_GATES)}")
    print(f"Training data: {config.DATA_PARAMS['default_ndata']}")
    print(f"Generations: {config.GA_PARAMS['ngen']}")
    print(f"Population: {config.GA_PARAMS['npop']}")
    print(f"Top solutions to save: {config.OUTPUT_PARAMS['save_top_n']}")
    print("\nUsage:")
    print("  python run.py --gates OR AND")
    print("  python run.py --all")
    print("  python run.py --interactive")
    print("  python run.py --config-summary")
    print("="*70 + "\n")


def parse_gates(gate_args):
    """Parse gate arguments from command line"""
    selected = []
    for gate_arg in gate_args:
        gate_name = gate_arg.upper()
        if gate_name in config.AVAILABLE_GATES:
            selected.append(config.AVAILABLE_GATES.index(gate_name))
    return selected


def interactive_selection():
    """Interactive gate selection"""
    print("\nAvailable gates:")
    for i, gate in enumerate(config.AVAILABLE_GATES):
        print(f"  {i}: {gate}")
    
    selection = input("\nEnter indices (e.g., '0 2 4') or 'all': ").strip()
    if selection.lower() == 'all':
        return list(range(len(config.AVAILABLE_GATES)))
    
    indices = [int(x) for x in selection.split()]
    return [i for i in indices if 0 <= i < len(config.AVAILABLE_GATES)]


# ==============================================================================
#                       VISUALIZATION FUNCTIONS
# ==============================================================================

def create_plots(gate_indices, results):
    """
    Create visualization plots for training, prediction, and truth tables
    
    Parameters:
    -----------
    gate_indices : list
        Indices of gates being plotted
    results : dict
        Dictionary containing results for each gate
    """
    # Training plot
    plt.figure(1, figsize=(15, 10))
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    # Prediction plot  
    plt.figure(2, figsize=(15, 10))
    
    # Truth table bar plot
    plt.figure(3, figsize=(15, 10))
    
    for idx, gate_idx in enumerate(gate_indices):
        gate_name = config.AVAILABLE_GATES[gate_idx]
        logic_data = results[gate_idx]['data']
        subplot_pos = idx + 1
        
        x = np.log10(logic_data[:, 0])
        y = np.log10(logic_data[:, 1])
        z_train = np.log10(logic_data[:, 2])
        z_pred = np.log10(logic_data[:, 3])
        
        # Training plot
        plt.figure(1)
        plt.subplot(2, 3, subplot_pos)
        plt.scatter(x, y, c=z_train, cmap='viridis', s=15, edgecolors='none')
        plt.colorbar()
        plt.xlabel('X1 (log10)')
        plt.ylabel('X2 (log10)')
        plt.title(f'{gate_name} - Training')
        
        # Prediction plot
        plt.figure(2)
        plt.subplot(2, 3, subplot_pos)
        plt.scatter(x, y, c=z_pred, cmap='viridis', s=20, edgecolors='none')
        plt.colorbar()
        plt.xlabel('X1 (log10)')
        plt.ylabel('X2 (log10)')
        plt.title(f'{gate_name} - Predicted')
        
        # Truth table bar plot
        bar_data = results[gate_idx]['bar_data']
        plt.figure(3)
        plt.subplot(2, 3, subplot_pos)
        bars = plt.bar(['00', '10', '01', '11'], bar_data['values'], color=bar_data['colors'])
        for bar, value in zip(bars, bar_data['values']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f"{value:.2f}", ha='center')
        plt.ylabel('Output (log10)')
        plt.title(f'{gate_name} - Truth Table')
    
    plt.figure(1)
    plt.tight_layout()
    plt.figure(2)
    plt.tight_layout()
    plt.figure(3)
    plt.tight_layout()


# ==============================================================================
#                       OPTIMIZATION AND TESTING
# ==============================================================================

def run_optimization(gate_indices):
    """
    Run optimization for selected gates
    
    Parameters:
    -----------
    gate_indices : list
        Indices of gates to optimize
        
    Returns:
    --------
    results : dict
        Dictionary containing results for each gate
    all_top_solutions : dict
        Dictionary containing top N solutions for each gate
    """
    ndata = config.DATA_PARAMS['default_ndata']
    ntest = config.DATA_PARAMS['default_ntest']
    
    print(f"\nGenerating {ndata} training and {ntest} test data...")
    
    # Generate data
    input_range = config.DATA_PARAMS['input_range']
    Xt2_train = np.random.uniform(*input_range, 2 * ndata)
    X_train = 10 ** Xt2_train.reshape(ndata, 2)
    
    Xt2_test = np.random.uniform(*input_range, 2 * ntest)
    X_test = 10 ** Xt2_test.reshape(ntest, 2)
    
    results = {}
    all_top_solutions = {}
    
    for ii in gate_indices:
        gate_name = config.AVAILABLE_GATES[ii]
        
        print(f"\n{'='*70}")
        print(f"Processing {gate_name}")
        print(f"{'='*70}")
        
        # Generate training data
        gate_func = datagen.get_gate_function(gate_name)
        on_cutoff = config.NN_PARAMS['on_cutoff']
        off_cutoff = config.NN_PARAMS['off_cutoff']
        Y_train = gate_func(X_train, on_cutoff, off_cutoff)
        
        # Run genetic algorithm
        log_w, best_topology, best_x_pos, best_y_pos, top_solutions = slg.run_genetic_algorithm(
            X_train, Y_train, ndata, config, gate_name, verbose=True
        )
        
        # Test network
        hidden_nodes = best_topology[:-1]
        output_node = best_topology[-1]
        network = slg.mlp(hidden_nodes, output_node)
        
        log_w = np.array(log_w)
        wH = log_w[0:2*len(hidden_nodes)].reshape(len(hidden_nodes), 2)
        wO = log_w[2*len(hidden_nodes):]
        
        # Generate predictions
        Y_test = gate_func(X_test, on_cutoff, off_cutoff)
        Y_pred = np.zeros(ntest)
        
        for i in range(ntest):
            Y_pred[i] = network.forward(X_test[i, :], wH, wO)
        
        logic_data = np.concatenate((
            X_test, Y_test.reshape(-1, 1), Y_pred.reshape(-1, 1)
        ), axis=1)
        
        # Evaluate on truth table
        bar_outputs = slg.evaluate_logic_gate(network, log_w, gate_name)
        bar_values = np.log10(bar_outputs)
        
        # Create color mapping for bar plot
        normalized = (bar_values - np.min(bar_values)) / (np.max(bar_values) - np.min(bar_values)) if np.max(bar_values) != np.min(bar_values) else np.zeros_like(bar_values)
        bar_colors = plt.cm.viridis(normalized)
        
        # Save best solution files
        base_name = config.OUTPUT_PARAMS['base_filename']
        np.save(f'{base_name}_weights_{gate_name}.npy', log_w)
        np.save(f'{base_name}_positions_{gate_name}.npy', [best_x_pos, best_y_pos])
        np.save(f'{base_name}_data_{gate_name}.npy', logic_data)
        
        # Store results
        results[ii] = {
            'data': logic_data,
            'topology': best_topology,
            'weights': log_w,
            'bar_data': {'values': bar_values, 'colors': bar_colors}
        }
        
        # Store top solutions
        all_top_solutions[gate_name] = top_solutions
        
        print(f"\nCompleted {gate_name}: {best_topology}")
    
    return results, all_top_solutions


def save_top_solutions(all_top_solutions, selected_gates):
    """
    Save top N performing solutions to file
    
    Parameters:
    -----------
    all_top_solutions : dict
        Dictionary of top solutions for each gate
    selected_gates : list
        List of gate names that were optimized
    """
    base_name = config.OUTPUT_PARAMS['base_filename']
    gates_str = "_".join(selected_gates)
    
    filename = f"{base_name}_top_solutions_{gates_str}.txt"
    
    with open(filename, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TOP PERFORMING SOLUTIONS\n")
        f.write("="*70 + "\n\n")
        
        for gate_name in selected_gates:
            f.write(f"\n{'='*70}\n")
            f.write(f"GATE: {gate_name}\n")
            f.write(f"{'='*70}\n\n")
            
            top_solutions = all_top_solutions[gate_name]
            
            for sol in top_solutions:
                f.write(f"Rank {sol['rank']}:\n")
                f.write(f"  Fitness: {sol['fitness']:.6f}\n")
                f.write(f"  Topology: {sol['topology']}\n")
                f.write(f"  X Positions: {sol['x_positions']}\n")
                f.write(f"  Y Positions: {sol['y_positions']}\n")
                f.write(f"  Weights: {sol['weights']}\n")
                f.write("\n" + "-"*70 + "\n\n")
    
    print(f"\nTop solutions saved to: {filename}")
    
    # Also save as numpy file for easy loading
    np_filename = f"{base_name}_top_solutions_{gates_str}.npy"
    np.save(np_filename, all_top_solutions, allow_pickle=True)
    print(f"Top solutions (numpy format) saved to: {np_filename}")


# ==============================================================================
#                       MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function"""
    
    if len(sys.argv) == 1:
        print_usage()
        return
    
    parser = argparse.ArgumentParser(description='Logic Gate Optimizer (2 Inputs)')
    parser.add_argument('--gates', nargs='+', help='Gate names')
    parser.add_argument('--all', action='store_true', help='Run all gates')
    parser.add_argument('--interactive', action='store_true', help='Interactive selection')
    parser.add_argument('--config-summary', action='store_true', help='Show config')
    
    args = parser.parse_args()
    
    if args.config_summary:
        config.print_config_summary()
        return
    
    if args.all:
        selected_indices = list(range(len(config.AVAILABLE_GATES)))
    elif args.interactive:
        selected_indices = interactive_selection()
    elif args.gates:
        selected_indices = parse_gates(args.gates)
    else:
        print("No gates selected.")
        return
    
    selected_gates = [config.AVAILABLE_GATES[i] for i in selected_indices]
    print(f"\nRunning gates: {selected_gates}")
    
    try:
        # Run optimization
        results, all_top_solutions = run_optimization(selected_indices)
        
        # Create visualization plots
        create_plots(selected_indices, results)
        
        # Save plots
        gates_str = "_".join(selected_gates)
        base_name = config.OUTPUT_PARAMS['base_filename']
        
        plt.figure(1)
        plt.savefig(f"{base_name}_training_{gates_str}.png", dpi=300, bbox_inches='tight')
        plt.figure(2)
        plt.savefig(f"{base_name}_prediction_{gates_str}.png", dpi=300, bbox_inches='tight')
        plt.figure(3)
        plt.savefig(f"{base_name}_truth_table_{gates_str}.png", dpi=300, bbox_inches='tight')
        
        # Save best topology
        with open(f"{base_name}_best_topology_{gates_str}.txt", 'w') as f:
            f.write("BEST NETWORK TOPOLOGIES\n")
            f.write("="*70 + "\n\n")
            for idx in selected_indices:
                gate_name = config.AVAILABLE_GATES[idx]
                topology = results[idx]['topology']
                f.write(f"{gate_name}: {topology}\n")
        
        # Save top N solutions
        save_top_solutions(all_top_solutions, selected_gates)
        
        print(f"\n{'='*70}")
        print("RESULTS SAVED")
        print(f"{'='*70}")
        print(f"Plots: {base_name}_training/prediction/truth_table_{gates_str}.png")
        print(f"Best topology: {base_name}_best_topology_{gates_str}.txt")
        print(f"Top {config.OUTPUT_PARAMS['save_top_n']} solutions: {base_name}_top_solutions_{gates_str}.txt/npy")
        print(f"Individual files: {base_name}_weights/positions/data_[GATE].npy")
        print(f"{'='*70}\n")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()

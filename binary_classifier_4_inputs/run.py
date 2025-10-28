# ==============================================================================
# RUN.PY - Main Execution Script for Binary Classifier (4 Inputs)
# ==============================================================================

import os
import numpy as np
from matplotlib import pyplot as plt
import argparse
import sys
import spatial_logic as slg
import data_generator as datagen
from config import Config

os.system('cls' if os.name == 'nt' else 'clear')

config = Config()

# ==============================================================================
#                       COMMAND LINE INTERFACE
# ==============================================================================

def print_usage():
    """Print usage information"""
    print("\n" + "="*70)
    print("BINARY CLASSIFIER WITH 4 INPUTS")
    print("="*70)
    print(f"Available operations: {', '.join(config.AVAILABLE_OPERATIONS)}")
    print(f"Data: 10 samples (0-9) with 4 binary inputs each")
    print(f"Generations: {config.GA_PARAMS['ngen']}")
    print(f"Population: {config.GA_PARAMS['npop']}")
    print(f"Top solutions to save: {config.OUTPUT_PARAMS['save_top_n']}")
    print("\nUsage:")
    print("  python run.py --ops Prime")
    print("  python run.py --ops Prime PerfectPower")
    print("  python run.py --all")
    print("  python run.py --config-summary")
    print("="*70 + "\n")

def parse_operations(op_args):
    """Parse operation arguments"""
    selected = []
    for op_arg in op_args:
        op_name = op_arg.capitalize()
        if op_name == 'Perfectpower':
            op_name = 'PerfectPower'
        if op_name in config.AVAILABLE_OPERATIONS:
            selected.append(config.AVAILABLE_OPERATIONS.index(op_name))
    return selected

# ==============================================================================
#                       VISUALIZATION FUNCTIONS
# ==============================================================================

def create_plots(op_indices, results):
    """Create visualization plots"""
    n_ops = len(op_indices)
    
    # Truth table plot
    plt.figure(1, figsize=(5*n_ops, 4))
    
    for idx, op_idx in enumerate(op_indices):
        op_name = config.AVAILABLE_OPERATIONS[op_idx]
        bar_data = results[op_idx]['bar_data']
        
        plt.subplot(1, n_ops, idx + 1)
        x_labels = [str(i) for i in range(10)]
        bars = plt.bar(x_labels, bar_data['values'], color=bar_data['colors'])
        
        for bar, value in zip(bars, bar_data['values']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{value:.2f}", ha='center', fontsize=8)
        
        plt.ylabel('Output (log10)')
        plt.xlabel('Number (0-9)')
        plt.title(f'{op_name} Classification')
        plt.ylim([0, max(bar_data['values']) * 1.2])
    
    plt.tight_layout()

# ==============================================================================
#                       OPTIMIZATION AND TESTING
# ==============================================================================

def run_optimization(op_indices):
    """Run optimization for selected operations"""
    ndata = config.DATA_PARAMS['ndata']
    on_cutoff = config.DATA_PARAMS['on_cutoff']
    off_cutoff = config.DATA_PARAMS['off_cutoff']
    
    print(f"\nGenerating data for {ndata} samples (0-9)...")
    
    # Create binary input data
    X_train, X2_train = datagen.create_binary_input_matrix(ndata, on_cutoff, off_cutoff)
    X_test, X2_test = datagen.create_binary_input_matrix(ndata, on_cutoff, off_cutoff)
    
    results = {}
    all_top_solutions = {}
    
    for ii in op_indices:
        op_name = config.AVAILABLE_OPERATIONS[ii]
        
        print(f"\n{'='*70}")
        print(f"Processing {op_name}")
        print(f"{'='*70}")
        
        # Generate training labels
        op_func = datagen.get_operation_function(op_name)
        Y_train = op_func(X2_train, on_cutoff, off_cutoff)
        
        # Run genetic algorithm
        log_w, best_topology, best_x_pos, best_y_pos, top_solutions = slg.run_genetic_algorithm(
            X_train, Y_train, ndata, config, op_name, verbose=True
        )
        
        # Test network
        hidden_nodes = best_topology[:-1]
        output_node = best_topology[-1]
        network = slg.mlp(hidden_nodes, output_node)
        
        # Evaluate on all numbers
        Y_pred = slg.evaluate_classifier(network, log_w, op_name)
        Y_test = op_func(X2_test, on_cutoff, off_cutoff)
        
        # Create bar plot data
        bar_values = np.log10(Y_pred)
        normalized = (bar_values - np.min(bar_values)) / (np.max(bar_values) - np.min(bar_values)) if np.max(bar_values) != np.min(bar_values) else np.zeros_like(bar_values)
        bar_colors = plt.cm.viridis(normalized)
        
        # Save files
        base_name = config.OUTPUT_PARAMS['base_filename']
        np.save(f'{base_name}_weights_{op_name}.npy', log_w)
        np.save(f'{base_name}_positions_{op_name}.npy', [best_x_pos, best_y_pos])
        np.save(f'{base_name}_predictions_{op_name}.npy', Y_pred)
        
        # Store results
        results[ii] = {
            'topology': best_topology,
            'weights': log_w,
            'predictions': Y_pred,
            'expected': Y_test,
            'bar_data': {'values': bar_values, 'colors': bar_colors}
        }
        
        all_top_solutions[op_name] = top_solutions
        
        print(f"\nCompleted {op_name}: {best_topology}")
        print(f"Predictions: {Y_pred}")
        print(f"Expected:    {Y_test}")
    
    return results, all_top_solutions

def save_top_solutions(all_top_solutions, selected_ops):
    """Save top N solutions to file"""
    base_name = config.OUTPUT_PARAMS['base_filename']
    ops_str = "_".join(selected_ops)
    
    filename = f"{base_name}_top_solutions_{ops_str}.txt"
    
    with open(filename, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TOP PERFORMING SOLUTIONS - BINARY CLASSIFIER\n")
        f.write("="*70 + "\n\n")
        
        for op_name in selected_ops:
            f.write(f"\n{'='*70}\n")
            f.write(f"OPERATION: {op_name}\n")
            f.write(f"{'='*70}\n\n")
            
            top_solutions = all_top_solutions[op_name]
            
            for sol in top_solutions:
                f.write(f"Rank {sol['rank']}:\n")
                f.write(f"  Fitness: {sol['fitness']:.6f}\n")
                f.write(f"  Topology: {sol['topology']}\n")
                f.write(f"  X Positions: {sol['x_positions']}\n")
                f.write(f"  Y Positions: {sol['y_positions']}\n")
                f.write(f"  Weights (first 10): {sol['weights'][:10]}\n")
                f.write("\n" + "-"*70 + "\n\n")
    
    print(f"\nTop solutions saved to: {filename}")
    
    # Save numpy format
    np_filename = f"{base_name}_top_solutions_{ops_str}.npy"
    np.save(np_filename, all_top_solutions, allow_pickle=True)
    print(f"Top solutions (numpy) saved to: {np_filename}")

# ==============================================================================
#                       MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function"""
    
    if len(sys.argv) == 1:
        print_usage()
        return
    
    parser = argparse.ArgumentParser(description='Binary Classifier (4 Inputs)')
    parser.add_argument('--ops', nargs='+', help='Operation names')
    parser.add_argument('--all', action='store_true', help='Run all operations')
    parser.add_argument('--config-summary', action='store_true', help='Show config')
    
    args = parser.parse_args()
    
    if args.config_summary:
        config.print_config_summary()
        return
    
    if args.all:
        selected_indices = list(range(len(config.AVAILABLE_OPERATIONS)))
    elif args.ops:
        selected_indices = parse_operations(args.ops)
    else:
        print("No operations selected.")
        return
    
    selected_ops = [config.AVAILABLE_OPERATIONS[i] for i in selected_indices]
    print(f"\nRunning operations: {selected_ops}")
    
    try:
        # Run optimization
        results, all_top_solutions = run_optimization(selected_indices)
        
        # Create plots
        create_plots(selected_indices, results)
        
        # Save plots
        ops_str = "_".join(selected_ops)
        base_name = config.OUTPUT_PARAMS['base_filename']
        
        plt.figure(1)
        plt.savefig(f"{base_name}_results_{ops_str}.png", dpi=300, bbox_inches='tight')
        
        # Save topology
        with open(f"{base_name}_best_topology_{ops_str}.txt", 'w') as f:
            f.write("BEST NETWORK TOPOLOGIES\n")
            f.write("="*70 + "\n\n")
            for idx in selected_indices:
                op_name = config.AVAILABLE_OPERATIONS[idx]
                topology = results[idx]['topology']
                f.write(f"{op_name}: {topology}\n")
        
        # Save top N solutions
        save_top_solutions(all_top_solutions, selected_ops)
        
        print(f"\n{'='*70}")
        print("RESULTS SAVED")
        print(f"{'='*70}")
        print(f"Plots: {base_name}_results_{ops_str}.png")
        print(f"Best topology: {base_name}_best_topology_{ops_str}.txt")
        print(f"Top {config.OUTPUT_PARAMS['save_top_n']} solutions: {base_name}_top_solutions_{ops_str}.txt/npy")
        print(f"Individual files: {base_name}_weights/positions/predictions_[OP].npy")
        print(f"{'='*70}\n")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()

# ==============================================================================
# DATA_GENERATOR.PY - Data Generation Functions for 2-Input Logic Gates
# ==============================================================================
# This file contains functions to generate training/test data for different
# 2-input logic gates: OR, AND, NOR, NAND, XOR
# ==============================================================================

import numpy as np

# ==============================================================================
#                       LOGIC GATE DATA GENERATORS
# ==============================================================================

def generate_OR(X, on_cutoff, off_cutoff):
    """
    Generate OR gate outputs
    OR: Output is HIGH if either input is HIGH
    Truth table: 00→0, 01→1, 10→1, 11→1
    """
    Y = np.zeros(X.shape[0])
    threshold = (on_cutoff + off_cutoff) / 2
    
    for i in range(X.shape[0]):
        if X[i, 0] > threshold or X[i, 1] > threshold:
            Y[i] = on_cutoff
        else:
            Y[i] = off_cutoff
    
    return Y


def generate_NOR(X, on_cutoff, off_cutoff):
    """
    Generate NOR gate outputs
    NOR: Output is HIGH only if both inputs are LOW
    Truth table: 00→1, 01→0, 10→0, 11→0
    """
    Y = np.zeros(X.shape[0])
    threshold = (on_cutoff + off_cutoff) / 2
    
    for i in range(X.shape[0]):
        if X[i, 0] <= threshold and X[i, 1] <= threshold:
            Y[i] = on_cutoff
        else:
            Y[i] = off_cutoff
    
    return Y


def generate_AND(X, on_cutoff, off_cutoff):
    """
    Generate AND gate outputs
    AND: Output is HIGH only if both inputs are HIGH
    Truth table: 00→0, 01→0, 10→0, 11→1
    """
    Y = np.zeros(X.shape[0])
    threshold = (on_cutoff + off_cutoff) / 2
    
    for i in range(X.shape[0]):
        if X[i, 0] > threshold and X[i, 1] > threshold:
            Y[i] = on_cutoff
        else:
            Y[i] = off_cutoff
    
    return Y


def generate_NAND(X, on_cutoff, off_cutoff):
    """
    Generate NAND gate outputs
    NAND: Output is LOW only if both inputs are HIGH
    Truth table: 00→1, 01→1, 10→1, 11→0
    """
    Y = np.zeros(X.shape[0])
    threshold = (on_cutoff + off_cutoff) / 2
    
    for i in range(X.shape[0]):
        if X[i, 0] > threshold and X[i, 1] > threshold:
            Y[i] = off_cutoff
        else:
            Y[i] = on_cutoff
    
    return Y


def generate_XOR(X, on_cutoff, off_cutoff):
    """
    Generate XOR gate outputs
    XOR: Output is HIGH if inputs are different
    Truth table: 00→0, 01→1, 10→1, 11→0
    """
    Y = np.zeros(X.shape[0])
    threshold = (on_cutoff + off_cutoff) / 2
    
    for i in range(X.shape[0]):
        input1_high = X[i, 0] > threshold
        input2_high = X[i, 1] > threshold
        
        if input1_high != input2_high:  # Inputs are different
            Y[i] = on_cutoff
        else:
            Y[i] = off_cutoff
    
    return Y


# ==============================================================================
#                       HELPER FUNCTIONS
# ==============================================================================

def get_gate_function(gate_name):
    """
    Return the appropriate data generation function for a gate name
    """
    gate_functions = {
        'OR': generate_OR,
        'NOR': generate_NOR,
        'AND': generate_AND,
        'NAND': generate_NAND,
        'XOR': generate_XOR
    }
    
    return gate_functions[gate_name]


def get_available_gates():
    """
    Return list of all available gate names
    """
    return ['OR', 'NOR', 'AND', 'NAND', 'XOR']

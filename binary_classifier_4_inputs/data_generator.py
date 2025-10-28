# ==============================================================================
# DATA_GENERATOR.PY - Data Generation for Binary Classifier (4 Inputs)
# ==============================================================================

import numpy as np

# ==============================================================================
#                       BINARY ENCODING FUNCTIONS
# ==============================================================================

def number_to_binary_4bit(n):
    """Convert number 0-9 to 4-bit binary representation"""
    return [int(bit) for bit in format(int(n), '04b')]


def create_binary_input_matrix(ndata, on_cutoff, off_cutoff):
    """
    Create input matrix with 4 binary inputs for numbers 0-9
    
    Parameters:
    -----------
    ndata : int
        Number of samples (should be 10 for 0-9)
    on_cutoff : float
        HIGH value for binary 1
    off_cutoff : float
        LOW value for binary 0
        
    Returns:
    --------
    X : array
        Binary input matrix (ndata, 4)
    X2 : array
        Original numbers 0-9
    """
    X2 = np.arange(ndata)
    X3 = np.array([number_to_binary_4bit(x) for x in X2])
    X = np.where(X3 == 0, off_cutoff, on_cutoff)
    return X, X2


# ==============================================================================
#                       PRIME NUMBER CLASSIFICATION
# ==============================================================================

def is_prime(n):
    """Check if number is prime"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def generate_Prime(X2, on_cutoff, off_cutoff):
    """
    Generate Prime number classification
    Prime numbers in 0-9: 2, 3, 5, 7
    
    Parameters:
    -----------
    X2 : array
        Original numbers (0-9)
    on_cutoff : float
        Output for prime = TRUE
    off_cutoff : float
        Output for not prime = FALSE
        
    Returns:
    --------
    Y : array
        Classification outputs
    """
    Y = np.zeros(len(X2))
    for i, num in enumerate(X2):
        Y[i] = on_cutoff if is_prime(int(num)) else off_cutoff
    return Y


# ==============================================================================
#                       PERFECT POWER CLASSIFICATION
# ==============================================================================

def is_perfect_power(n):
    """
    Check if number is a perfect power (a^b where b >= 2)
    Perfect powers in 0-9: 0, 1, 4, 8, 9
    """
    if n == 0 or n == 1:
        return True
    
    # Check for perfect squares, cubes, etc.
    for exp in range(2, int(np.log2(n)) + 2):
        base = n ** (1.0 / exp)
        if abs(base - round(base)) < 1e-9:
            return True
    return False


def generate_PerfectPower(X2, on_cutoff, off_cutoff):
    """
    Generate Perfect Power classification
    Perfect powers in 0-9: 0, 1, 4 (2²), 8 (2³), 9 (3²)
    
    Parameters:
    -----------
    X2 : array
        Original numbers (0-9)
    on_cutoff : float
        Output for perfect power = TRUE
    off_cutoff : float
        Output for not perfect power = FALSE
        
    Returns:
    --------
    Y : array
        Classification outputs
    """
    Y = np.zeros(len(X2))
    for i, num in enumerate(X2):
        Y[i] = on_cutoff if is_perfect_power(int(num)) else off_cutoff
    return Y


# ==============================================================================
#                       VOWEL CLASSIFICATION
# ==============================================================================

def is_vowel(n):
    """
    Check if number represents a vowel
    Mapping: 0->A, 1->B, 2->C, 3->D, 4->E, 5->F, 6->G, 7->H, 8->I, 9->J
    Vowels: A(0), E(4), I(8)
    """
    return n in [0, 4, 8]


def generate_Vowel(X2, on_cutoff, off_cutoff):
    """
    Generate Vowel classification
    Vowels when mapped to letters: A(0), E(4), I(8)
    
    Parameters:
    -----------
    X2 : array
        Original numbers (0-9)
    on_cutoff : float
        Output for vowel = TRUE
    off_cutoff : float
        Output for not vowel = FALSE
        
    Returns:
    --------
    Y : array
        Classification outputs
    """
    Y = np.zeros(len(X2))
    for i, num in enumerate(X2):
        Y[i] = on_cutoff if is_vowel(int(num)) else off_cutoff
    return Y


# ==============================================================================
#                       HELPER FUNCTIONS
# ==============================================================================

def get_operation_function(operation_name):
    """
    Return the appropriate data generation function
    
    Parameters:
    -----------
    operation_name : str
        Name of operation ('Prime', 'PerfectPower', or 'Vowel')
        
    Returns:
    --------
    function
        Corresponding data generation function
    """
    operations = {
        'Prime': generate_Prime,
        'PerfectPower': generate_PerfectPower,
        'Vowel': generate_Vowel
    }
    return operations[operation_name]


def get_available_operations():
    """Return list of all available operations"""
    return ['Prime', 'PerfectPower', 'Vowel']

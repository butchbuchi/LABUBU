"""Steane interaction-layer expansion for entanglement-cost scheduling."""
import numpy as np
def g1(ft_layers, target_layer=[]):
    for i in range(int(ft_layers.shape[0] / 8)):
        ft_layers[i * 8 + 7, target_layer[2]] = i * 8 + 0
        ft_layers[i * 8 + 0, target_layer[2]] = i * 8 + 7
        ft_layers[i * 8 + 7, target_layer[3]] = i * 8 + 4
        ft_layers[i * 8 + 4, target_layer[3]] = i * 8 + 7
        ft_layers[i * 8 + 7, target_layer[4]] = i * 8 + 5
        ft_layers[i * 8 + 5, target_layer[4]] = i * 8 + 7
        ft_layers[i * 8 + 7, target_layer[5]] = i * 8 + 6
        ft_layers[i * 8 + 6, target_layer[5]] = i * 8 + 7

def g2(ft_layers, target_layer=[]):
    for i in range(int(ft_layers.shape[0] / 8)):
        ft_layers[i * 8 + 7, target_layer[2]] = i * 8 + 1
        ft_layers[i * 8 + 1, target_layer[2]] = i * 8 + 7
        ft_layers[i * 8 + 7, target_layer[3]] = i * 8 + 3
        ft_layers[i * 8 + 3, target_layer[3]] = i * 8 + 7
        ft_layers[i * 8 + 7, target_layer[4]] = i * 8 + 5
        ft_layers[i * 8 + 5, target_layer[4]] = i * 8 + 7
        ft_layers[i * 8 + 7, target_layer[5]] = i * 8 + 6
        ft_layers[i * 8 + 6, target_layer[5]] = i * 8 + 7

def g3(ft_layers, target_layer=[]):
    for i in range(int(ft_layers.shape[0] / 8)):
        ft_layers[i * 8 + 7, target_layer[2]] = i * 8 + 2
        ft_layers[i * 8 + 2, target_layer[2]] = i * 8 + 7
        ft_layers[i * 8 + 7, target_layer[3]] = i * 8 + 3
        ft_layers[i * 8 + 3, target_layer[3]] = i * 8 + 7
        ft_layers[i * 8 + 7, target_layer[4]] = i * 8 + 4
        ft_layers[i * 8 + 4, target_layer[4]] = i * 8 + 7
        ft_layers[i * 8 + 7, target_layer[5]] = i * 8 + 5
        ft_layers[i * 8 + 5, target_layer[5]] = i * 8 + 7

def g4(ft_layers, target_layer=[]):
    for i in range(int(ft_layers.shape[0] / 8)):
        ft_layers[i * 8 + 7, target_layer[2]] = i * 8 + 0
        ft_layers[i * 8 + 0, target_layer[2]] = i * 8 + 7
        ft_layers[i * 8 + 7, target_layer[4]] = i * 8 + 4
        ft_layers[i * 8 + 4, target_layer[4]] = i * 8 + 7
        ft_layers[i * 8 + 7, target_layer[6]] = i * 8 + 5
        ft_layers[i * 8 + 5, target_layer[6]] = i * 8 + 7
        ft_layers[i * 8 + 7, target_layer[8]] = i * 8 + 6
        ft_layers[i * 8 + 6, target_layer[8]] = i * 8 + 7

def g5(ft_layers, target_layer=[]):
    for i in range(int(ft_layers.shape[0] / 8)):
        ft_layers[i * 8 + 7, target_layer[2]] = i * 8 + 1
        ft_layers[i * 8 + 1, target_layer[2]] = i * 8 + 7
        ft_layers[i * 8 + 7, target_layer[4]] = i * 8 + 3
        ft_layers[i * 8 + 3, target_layer[4]] = i * 8 + 7
        ft_layers[i * 8 + 7, target_layer[6]] = i * 8 + 5
        ft_layers[i * 8 + 5, target_layer[6]] = i * 8 + 7
        ft_layers[i * 8 + 7, target_layer[8]] = i * 8 + 6
        ft_layers[i * 8 + 6, target_layer[8]] = i * 8 + 7

def g6(ft_layers, target_layer=[]):
    for i in range(int(ft_layers.shape[0] / 8)):
        ft_layers[i * 8 + 7, target_layer[2]] = i * 8 + 2
        ft_layers[i * 8 + 2, target_layer[2]] = i * 8 + 7
        ft_layers[i * 8 + 7, target_layer[4]] = i * 8 + 3
        ft_layers[i * 8 + 3, target_layer[4]] = i * 8 + 7
        ft_layers[i * 8 + 7, target_layer[6]] = i * 8 + 4
        ft_layers[i * 8 + 4, target_layer[6]] = i * 8 + 7
        ft_layers[i * 8 + 7, target_layer[8]] = i * 8 + 5
        ft_layers[i * 8 + 5, target_layer[8]] = i * 8 + 7

def syndrome_circuit(ft_layers, target_layer=[]):
    """
    Constructs the syndrome circuit for the FT circuit.
    
    Parameters:
    ft_layers (np.ndarray): The ft_layers of the original circuit.
    target_layer (list): The target layer indices for the syndrome circuit.
    
    Returns:
    np.ndarray: The modified ft_layers with syndrome operations applied.
    """
    g1(ft_layers, target_layer[0:8])
    g2(ft_layers, target_layer[8:16])
    g3(ft_layers, target_layer[16:24])
    g4(ft_layers, target_layer[24:35])
    g5(ft_layers, target_layer[35:46])
    g6(ft_layers, target_layer[46:57])
    return ft_layers

def initialize_ft_circuit(ft_layers):
    g1(ft_layers, list(range(0, 8)))
    g2(ft_layers, list(range(8, 16)))
    g3(ft_layers, list(range(16, 24)))
    return ft_layers

def FT_circuit_func(layers, period):
    layers = layers.T
    origin_depth = layers.shape[-1]
    origin_width = layers.shape[0]
    ft_depth = int(25 + np.ceil(origin_depth / period) * (3 * 8 + 3 * 11 + 1) + origin_depth)
    ft_width = int(origin_width * 8)
    ft_layers = -1 * np.ones((ft_width, ft_depth), dtype=int)
    ft_layers = initialize_ft_circuit(ft_layers)
    interval = 25
    for i in range(origin_depth):
        for j in range(origin_width):
            for k in range(7):
                if layers[j, i] != -1:
                    ft_layers[8 * j + k, i + interval] = layers[j, i] * 8 + k
        if (i + 1) % period == 0:
            ft_layers = syndrome_circuit(ft_layers, list(range(i + interval + 1, i + interval + 1 + 58 + 1)))
            interval += 58
    if origin_depth % period != 0:
        ft_layers = syndrome_circuit(ft_layers, list(range(origin_depth + interval, origin_depth + interval + 58 + 1)))
    return ft_layers.T

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:54:14 2026

@author: Owner
"""

import numpy as np
import jax
import jax.numpy as jnp
from deepmd.infer import DeepPot

# Enable 64-bit precision (critical for accurate second-order derivatives)
jax.config.update("jax_enable_x64", True)

def calculate_force_constants(model_path, coordinates, box_cell, atom_types):
    """
    Loads a JAX-compatible DeePMD model and evaluates the full Hessian matrix 
    to extract atomic force constants.
    """
    # 1. Initialize the Deep Potential with the JAX backend
    dp = DeepPot(model_path)
    
    # Ensure variables are mapped into JAX arrays
    # Coordinates shape must be (1, N * 3) for standard DeepPot inference
    coord_jax = jnp.array(coordinates, dtype=jnp.float64).reshape(1, -1)
    cell_jax = jnp.array(box_cell, dtype=jnp.float64).reshape(1, -1)
    atype_jax = np.array(atom_types, dtype=np.int32)
    
    # 2. Define a clean function wrapper that only returns energy
    def energy_fn(coords):
        # dp.eval returns a tuple: (energy, forces, virial)
        # We explicitly index [0][0] to retrieve the scalar energy value
        return dp.eval(coords, cell_jax, atype_jax)[0][0]
        
    # 3. Compute the analytical Hessian using JAX autodiff
    # jacfwd(jacrev(energy_fn)) could also be used for specific memory profiles
    hessian_fn = jax.hessian(energy_fn)
    
    # Evaluates the matrix of size (3N, 3N)
    raw_hessian = hessian_fn(coord_jax)
    
    # 4. Clean up dimensions to shape (3N, 3N)
    hessian_matrix = jnp.squeeze(raw_hessian)
    
    # 5. Convert to force constants (Interatomic Force Constant matrix is -Hessian)
    force_constants = -hessian_matrix
    
    return force_constants

# ==========================================
# Example Batch Workflow Execution
# ==========================================
if __name__ == "__main__":
    # Define a list of your converted JAX models
    potential_files = [
        "bulk_system.savedmodel", 
        "interface_system.savedmodel"
    ]
    
    # Dummy system configuration (3 atoms as an example)
    # Replace these with your actual bulk/interface configuration data
    coords = np.array([[0.0, 0.0, 0.0], 
                       [1.2, 0.0, 0.0], 
                       [0.6, 1.0, 0.0]])
    
    cell = np.array([[10.0, 0.0, 0.0], 
                     [0.0, 10.0, 0.0], 
                     [0.0, 0.0, 10.0]])
    
    atom_types = [0, 1, 0] # Internal DeePMD atom type map integers
    
    # Iterate and calculate matrices for each model
    for model in potential_files:
        print(f"Processing model: {model}")
        fc_matrix = calculate_force_constants(model, coords, cell, atom_types)
        print(f"Force Constant Matrix Shape: {fc_matrix.shape}")
        
        # Save output matrices to disk
        np.save(f"fc_matrix_{model.split('.')[0]}.npy", fc_matrix)

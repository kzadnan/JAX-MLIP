# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 18:09:31 2026

@author: Owner
"""

import numpy as np

# 1. Load your generated Hessian matrix
H_analytical = np.load("hessian_analytical_output.npy")
H_fd=np.load("hessian_finite_difference_matrix.npy")


error=H_analytical-H_fd
IFC= np.load("ifc_matrix.npy")
n_dof = H_analytical.shape[0]
natoms = n_dof // 3

# 2. Compute the sum along each row (axis=1)
# For a perfect ASR, row_sums should be an array of zeros.
row_sums = np.sum(H_analytical, axis=1)

# 3. Print out the maximum deviation from zero
max_asr_deviation = np.max(np.abs(row_sums))

print("=== Acoustic Sum Rule (ASR) Check ===")
print(f"Total atoms in system        : {natoms}")
print(f"Total degrees of freedom     : {n_dof}")
print(f"Maximum raw ASR row deviation: {max_asr_deviation:.6e} eV/Å²")

# 4. View a snapshot of the raw row sums to see the directional errors
print("\nFirst 6 row sums (Atom 0 and Atom 1 x,y,z components):")
print(row_sums[:6])
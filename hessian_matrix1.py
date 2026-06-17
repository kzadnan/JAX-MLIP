"""
Analytical Hessian Generator from a LAMMPS Data File
===================================================
Strategy: Automatically parses structural geometries, cell vectors, and 
atom types from a 'lammps.data' file using ASE, then pipes them into 
the robust DeePMD force-displacement evaluator.
"""

import numpy as np
from deepmd.infer import DeepPot
from ase.io import read

def calculate_hessian_via_forces(model_path: str,
                                coordinates: np.ndarray,
                                box_cell: np.ndarray or None,
                                atom_types: list or np.ndarray,
                                eps: float = 1e-3) -> dict:
    """
    Compute the Hessian matrix by displacing coordinates and analyzing force deviations.
    """
    dp = DeepPot(model_path, backend="jax")
    
    coords_base = np.asarray(coordinates, dtype=np.float64)
    natoms = coords_base.shape[0]
    n_dof = 3 * natoms
    
    coords_flat = coords_base.reshape(-1).copy()
    box_np = np.asarray(box_cell, dtype=np.float64).reshape(1, 9) if box_cell is not None else None
    atype_list = list(atom_types)

    e0, f0, v0, *_ = dp.eval(coords_base.reshape(1, natoms, 3), box_np, atype_list)
    H = np.zeros((n_dof, n_dof), dtype=np.float64)
    
    print(f"Computing Hessian Matrix ({n_dof}x{n_dof}) via force shifts (eps = {eps} Å)...")

    for j in range(n_dof):
        r_plus = coords_flat.copy()
        r_plus[j] += eps
        _, f_plus, *_ = dp.eval(r_plus.reshape(1, natoms, 3), box_np, atype_list)
        
        r_minus = coords_flat.copy()
        r_minus[j] -= eps
        _, f_minus, *_ = dp.eval(r_minus.reshape(1, natoms, 3), box_np, atype_list)
        
        df_drj = (f_plus.reshape(-1) - f_minus.reshape(-1)) / (2.0 * eps)
        H[:, j] = -df_drj

    H = 0.5 * (H + H.T)

    return {
        "energy":         np.squeeze(e0),
        "force":          np.squeeze(f0),
        "virial":         np.squeeze(v0) if v0 is not None else np.zeros((3,3)),
        "hessian":        H,
        "force_jacobian": -H
    }


# Execution Entry Block
if __name__ == "__main__":
    # 1. Define paths
    MY_MODEL_PATH = "graph-compress.pb"
    LAMMPS_DATA_PATH = "C_unitcell_new.txt"
    
    try:
        print(f"Parsing structure file from: {LAMMPS_DATA_PATH}")
        # 2. Load the structure using ASE
        # format='lammps-data' natively extracts positions, box boundaries, and atom IDs
        atoms = read(LAMMPS_DATA_PATH, format='lammps-data')
        
        # 3. Extract the required arrays for DeePMD
        real_coords = atoms.get_positions()                # Shape: (N, 3)
        real_box    = atoms.get_cell().flat               # Shape: (9,)
        
        # 4. Map atom categories to 0-indexed integers
        # DeePMD requires type maps starting at 0, 1, 2... 
        # Since LAMMPS type indices start at 1, we subtract 1 to align cleanly.
        real_types = np.array(atoms.get_array('type'), dtype=np.int32) - 1
        
        print(f"Structure loaded successfully. Total atoms detected: {len(atoms)}")
        print(f"Unique atom types detected (0-indexed): {np.unique(real_types)}")
        
        # 5. Evaluate the analytical matrix
        results = calculate_hessian_via_forces(MY_MODEL_PATH, real_coords, real_box, real_types)
        
        print("\n--- ✅ Evaluation Successful ---")
        print("Hessian Matrix Shape:", results["hessian"].shape)
        
        # Optional: Save your Hessian matrix to a binary file for easy post-processing
        np.save("hessian_analytical_output.npy", results["hessian"])
        print("Hessian matrix exported successfully to 'hessian_analytical_output.npy'")
        
    except Exception as e:
        print(f"\n--- ❌ Evaluation Failed ---\nError Type: {type(e).__name__}\nDetails: {e}")

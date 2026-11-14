"""
Molecular Docking Tool for Virtual Screening

This tool performs structure-based virtual screening via molecular docking of ligands 
to protein targets using AutoDock Vina.
"""

import os
import subprocess
import tempfile
import logging
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import json

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
except ImportError:
    Chem = None
    AllChem = None
    Descriptors = None

try:
    from vina import Vina
except ImportError:
    Vina = None

from smolagents import tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@tool
def molecular_docking_tool(
    protein_pdb_path: str,
    ligand_smiles: Union[str, List[str]],
    center_x: float,
    center_y: float,
    center_z: float,
    size_x: float = 20.0,
    size_y: float = 20.0,
    size_z: float = 20.0,
    exhaustiveness: int = 8,
    num_modes: int = 9,
    energy_range: float = 3.0,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Perform molecular docking of ligands to a protein target using AutoDock Vina.
    
    This tool conducts structure-based virtual screening by docking small molecule
    ligands to protein binding sites and returns binding scores and poses.
    
    Args:
        protein_pdb_path (str): Path to the protein PDB file
        ligand_smiles (Union[str, List[str]]): SMILES string(s) of ligand(s) to dock
        center_x (float): X coordinate of the docking box center
        center_y (float): Y coordinate of the docking box center
        center_z (float): Z coordinate of the docking box center
        size_x (float, optional): Size of docking box in X dimension. Defaults to 20.0.
        size_y (float, optional): Size of docking box in Y dimension. Defaults to 20.0.
        size_z (float, optional): Size of docking box in Z dimension. Defaults to 20.0.
        exhaustiveness (int, optional): Exhaustiveness of search. Defaults to 8.
        num_modes (int, optional): Number of binding modes to generate. Defaults to 9.
        energy_range (float, optional): Energy range for binding modes. Defaults to 3.0.
        output_dir (Optional[str], optional): Output directory for results. Defaults to None.
    
    Returns:
        Dict: Dictionary containing docking results with binding scores, poses, and metadata
    
    Raises:
        ImportError: If required dependencies are not installed
        FileNotFoundError: If protein PDB file is not found
        ValueError: If invalid inputs are provided
    """
    
    # Validate dependencies
    if Chem is None or AllChem is None:
        raise ImportError("RDKit is required but not installed. Install with: conda install -c conda-forge rdkit")
    
    if Vina is None:
        raise ImportError("AutoDock Vina Python bindings are required. Install with: pip install vina")
    
    # Validate inputs
    if not os.path.exists(protein_pdb_path):
        raise FileNotFoundError(f"Protein PDB file not found: {protein_pdb_path}")
    
    if not protein_pdb_path.lower().endswith('.pdb'):
        raise ValueError("Protein file must be in PDB format")
    
    # Ensure ligand_smiles is a list
    if isinstance(ligand_smiles, str):
        ligand_smiles = [ligand_smiles]
    
    # Validate SMILES strings
    for smiles in ligand_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
    
    # Set up output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="molecular_docking_")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting molecular docking with {len(ligand_smiles)} ligand(s)")
    logger.info(f"Output directory: {output_dir}")
    
    results = {
        "protein_file": protein_pdb_path,
        "docking_box": {
            "center": [center_x, center_y, center_z],
            "size": [size_x, size_y, size_z]
        },
        "parameters": {
            "exhaustiveness": exhaustiveness,
            "num_modes": num_modes,
            "energy_range": energy_range
        },
        "ligands": [],
        "output_directory": output_dir,
        "status": "success"
    }
    
    try:
        # Process each ligand
        for i, smiles in enumerate(ligand_smiles):
            logger.info(f"Processing ligand {i+1}/{len(ligand_smiles)}: {smiles}")
            
            ligand_result = _process_single_ligand(
                smiles=smiles,
                protein_pdb_path=protein_pdb_path,
                center=(center_x, center_y, center_z),
                size=(size_x, size_y, size_z),
                exhaustiveness=exhaustiveness,
                num_modes=num_modes,
                energy_range=energy_range,
                output_dir=output_dir,
                ligand_id=f"ligand_{i+1}"
            )
            
            results["ligands"].append(ligand_result)
        
        # Save results to JSON file
        results_file = os.path.join(output_dir, "docking_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Docking completed successfully. Results saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Error during docking: {str(e)}")
        results["status"] = "error"
        results["error_message"] = str(e)
    
    return results


def _process_single_ligand(
    smiles: str,
    protein_pdb_path: str,
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    exhaustiveness: int,
    num_modes: int,
    energy_range: float,
    output_dir: str,
    ligand_id: str
) -> Dict:
    """
    Process a single ligand for docking.
    
    Args:
        smiles (str): SMILES string of the ligand
        protein_pdb_path (str): Path to protein PDB file
        center (Tuple[float, float, float]): Docking box center coordinates
        size (Tuple[float, float, float]): Docking box size
        exhaustiveness (int): Search exhaustiveness
        num_modes (int): Number of binding modes
        energy_range (float): Energy range for modes
        output_dir (str): Output directory
        ligand_id (str): Unique identifier for the ligand
    
    Returns:
        Dict: Docking results for the ligand
    """
    
    ligand_result = {
        "ligand_id": ligand_id,
        "smiles": smiles,
        "binding_scores": [],
        "poses": [],
        "molecular_properties": {},
        "status": "success"
    }
    
    try:
        # Generate 3D structure from SMILES
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        
        # Generate conformer
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        # Calculate molecular properties
        ligand_result["molecular_properties"] = {
            "molecular_weight": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "hbd": Descriptors.NumHDonors(mol),
            "hba": Descriptors.NumHAcceptors(mol),
            "tpsa": Descriptors.TPSA(mol),
            "rotatable_bonds": Descriptors.NumRotatableBonds(mol)
        }
        
        # Save ligand as SDF file
        ligand_sdf_path = os.path.join(output_dir, f"{ligand_id}.sdf")
        writer = Chem.SDWriter(ligand_sdf_path)
        writer.write(mol)
        writer.close()
        
        # Convert to PDBQT format for Vina
        ligand_pdbqt_path = os.path.join(output_dir, f"{ligand_id}.pdbqt")
        _convert_sdf_to_pdbqt(ligand_sdf_path, ligand_pdbqt_path)
        
        # Prepare protein PDBQT if needed
        protein_pdbqt_path = os.path.join(output_dir, "protein.pdbqt")
        if not os.path.exists(protein_pdbqt_path):
            _convert_pdb_to_pdbqt(protein_pdb_path, protein_pdbqt_path)
        
        # Perform docking with Vina
        v = Vina(sf_name='vina')
        v.set_receptor(protein_pdbqt_path)
        v.set_ligand_from_file(ligand_pdbqt_path)
        v.compute_vina_maps(center=center, box_size=size)
        
        # Run docking
        v.dock(exhaustiveness=exhaustiveness, n_poses=num_modes)
        
        # Get results
        output_pdbqt_path = os.path.join(output_dir, f"{ligand_id}_docked.pdbqt")
        v.write_poses(output_pdbqt_path, n_poses=num_modes, overwrite=True)
        
        # Extract binding scores
        scores = v.score()
        for j, score in enumerate(scores):
            if j < num_modes and score <= scores[0] + energy_range:
                ligand_result["binding_scores"].append(score)
                ligand_result["poses"].append({
                    "pose_id": j + 1,
                    "binding_affinity": score,
                    "file_path": output_pdbqt_path
                })
        
        logger.info(f"Docking completed for {ligand_id}. Best score: {scores[0]:.2f} kcal/mol")
        
    except Exception as e:
        logger.error(f"Error processing ligand {ligand_id}: {str(e)}")
        ligand_result["status"] = "error"
        ligand_result["error_message"] = str(e)
    
    return ligand_result


def _convert_sdf_to_pdbqt(sdf_path: str, pdbqt_path: str) -> None:
    """
    Convert SDF file to PDBQT format using Open Babel.
    
    Args:
        sdf_path (str): Input SDF file path
        pdbqt_path (str): Output PDBQT file path
    """
    try:
        cmd = f"obabel {sdf_path} -O {pdbqt_path} -p 7.4"
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        # Fallback: create a simple PDBQT conversion
        logger.warning("Open Babel not available, using simplified conversion")
        _simple_sdf_to_pdbqt_conversion(sdf_path, pdbqt_path)


def _convert_pdb_to_pdbqt(pdb_path: str, pdbqt_path: str) -> None:
    """
    Convert PDB file to PDBQT format.
    
    Args:
        pdb_path (str): Input PDB file path
        pdbqt_path (str): Output PDBQT file path
    """
    try:
        cmd = f"prepare_receptor4.py -r {pdb_path} -o {pdbqt_path}"
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        # Fallback: simple conversion
        logger.warning("AutoDockTools not available, using simplified conversion")
        _simple_pdb_to_pdbqt_conversion(pdb_path, pdbqt_path)


def _simple_sdf_to_pdbqt_conversion(sdf_path: str, pdbqt_path: str) -> None:
    """
    Simple SDF to PDBQT conversion fallback.
    
    Args:
        sdf_path (str): Input SDF file path
        pdbqt_path (str): Output PDBQT file path
    """
    # This is a simplified conversion - in production, use proper tools
    with open(sdf_path, 'r') as f:
        sdf_content = f.read()
    
    # Basic conversion (this is simplified and may not work for all cases)
    pdbqt_content = sdf_content.replace('M  END', 'TORSDOF 0')
    
    with open(pdbqt_path, 'w') as f:
        f.write(pdbqt_content)


def _simple_pdb_to_pdbqt_conversion(pdb_path: str, pdbqt_path: str) -> None:
    """
    Simple PDB to PDBQT conversion fallback.
    
    Args:
        pdb_path (str): Input PDB file path
        pdbqt_path (str): Output PDBQT file path
    """
    # This is a simplified conversion - in production, use proper tools
    with open(pdb_path, 'r') as f:
        pdb_content = f.read()
    
    # Basic conversion (remove water and add charges)
    lines = pdb_content.split('\n')
    pdbqt_lines = []
    
    for line in lines:
        if line.startswith('ATOM') and 'HOH' not in line:
            # Add partial charges (simplified)
            pdbqt_lines.append(line.rstrip() + '    0.000')
        elif line.startswith('HETATM') and 'HOH' not in line:
            pdbqt_lines.append(line.rstrip() + '    0.000')
    
    with open(pdbqt_path, 'w') as f:
        f.write('\n'.join(pdbqt_lines))


if __name__ == "__main__":
    # Example usage
    print("Molecular Docking Tool")
    print("This tool requires:")
    print("- RDKit: conda install -c conda-forge rdkit")
    print("- AutoDock Vina: pip install vina")
    print("- Optional: Open Babel for file conversion")
    print("- Optional: AutoDockTools for advanced preparation")
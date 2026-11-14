"""
Ligand-based virtual screening tool for molecular similarity searches and pharmacophore modeling.

This tool performs virtual screening using molecular fingerprints, similarity calculations,
and property filtering to identify compounds similar to a reference ligand.
"""

import os
import sys
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, DataStructs
    from rdkit.Chem.Fingerprints import FingerprintMols
    from rdkit.Chem.AtomPairs import Pairs
    from rdkit.Chem.Pharm2D.SigFactory import SigFactory
    from rdkit.Chem.Pharm2D import Generate
    from rdkit.Chem.Features import FeatureParser
    from rdkit import RDLogger
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Required dependencies not found: {e}")
    print("Please install: pip install rdkit pandas numpy")
    sys.exit(1)

from smolagents import tool

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LigandBasedScreener:
    """
    A comprehensive ligand-based virtual screening tool.
    """
    
    def __init__(self):
        """Initialize the screening tool with default parameters."""
        self.fingerprint_types = {
            'morgan': self._get_morgan_fp,
            'rdkit': self._get_rdkit_fp,
            'atom_pairs': self._get_atom_pairs_fp,
            'topological_torsion': self._get_topological_torsion_fp,
            'maccs': self._get_maccs_fp
        }
        
        # Property filters (typical drug-like ranges)
        self.default_filters = {
            'mw': (150, 500),  # Molecular weight
            'logp': (-2, 5),   # LogP
            'hbd': (0, 5),     # H-bond donors
            'hba': (0, 10),    # H-bond acceptors
            'rotatable_bonds': (0, 10),
            'tpsa': (0, 140)   # Topological polar surface area
        }
    
    def _get_morgan_fp(self, mol: Chem.Mol, radius: int = 2, nbits: int = 2048) -> DataStructs.ExplicitBitVect:
        """Generate Morgan fingerprint."""
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    
    def _get_rdkit_fp(self, mol: Chem.Mol, nbits: int = 2048) -> DataStructs.ExplicitBitVect:
        """Generate RDKit fingerprint."""
        return FingerprintMols.FingerprintMol(mol, fpSize=nbits)
    
    def _get_atom_pairs_fp(self, mol: Chem.Mol) -> DataStructs.IntSparseIntVect:
        """Generate atom pairs fingerprint."""
        return Pairs.GetAtomPairFingerprint(mol)
    
    def _get_topological_torsion_fp(self, mol: Chem.Mol) -> DataStructs.LongSparseIntVect:
        """Generate topological torsion fingerprint."""
        return rdMolDescriptors.GetHashedTopologicalTorsionFingerprint(mol)
    
    def _get_maccs_fp(self, mol: Chem.Mol) -> DataStructs.ExplicitBitVect:
        """Generate MACCS keys fingerprint."""
        return rdMolDescriptors.GetMACCSKeysFingerprint(mol)
    
    def _calculate_properties(self, mol: Chem.Mol) -> Dict[str, float]:
        """Calculate molecular properties."""
        try:
            return {
                'mw': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'tpsa': Descriptors.TPSA(mol),
                'num_atoms': mol.GetNumAtoms(),
                'num_rings': Descriptors.RingCount(mol)
            }
        except Exception as e:
            logger.warning(f"Error calculating properties: {e}")
            return {}
    
    def _apply_filters(self, mol: Chem.Mol, filters: Optional[Dict[str, Tuple[float, float]]] = None) -> bool:
        """Apply property filters to a molecule."""
        if filters is None:
            filters = self.default_filters
        
        properties = self._calculate_properties(mol)
        
        for prop, (min_val, max_val) in filters.items():
            if prop in properties:
                value = properties[prop]
                if not (min_val <= value <= max_val):
                    return False
        return True
    
    def _calculate_similarity(self, fp1, fp2, metric: str = 'tanimoto') -> float:
        """Calculate similarity between two fingerprints."""
        try:
            if metric.lower() == 'tanimoto':
                return DataStructs.TanimotoSimilarity(fp1, fp2)
            elif metric.lower() == 'dice':
                return DataStructs.DiceSimilarity(fp1, fp2)
            elif metric.lower() == 'cosine':
                return DataStructs.CosineSimilarity(fp1, fp2)
            else:
                logger.warning(f"Unknown similarity metric: {metric}. Using Tanimoto.")
                return DataStructs.TanimotoSimilarity(fp1, fp2)
        except Exception as e:
            logger.warning(f"Error calculating similarity: {e}")
            return 0.0
    
    def screen_compounds(
        self,
        query_smiles: str,
        compound_library: List[str],
        fingerprint_type: str = 'morgan',
        similarity_threshold: float = 0.7,
        max_results: int = 100,
        similarity_metric: str = 'tanimoto',
        apply_filters: bool = True,
        custom_filters: Optional[Dict[str, Tuple[float, float]]] = None,
        include_3d: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Perform ligand-based virtual screening.
        
        Args:
            query_smiles: SMILES string of the reference ligand
            compound_library: List of SMILES strings to screen
            fingerprint_type: Type of fingerprint ('morgan', 'rdkit', 'atom_pairs', 'topological_torsion', 'maccs')
            similarity_threshold: Minimum similarity score (0.0-1.0)
            max_results: Maximum number of results to return
            similarity_metric: Similarity metric ('tanimoto', 'dice', 'cosine')
            apply_filters: Whether to apply drug-like property filters
            custom_filters: Custom property filters as {property: (min, max)}
            include_3d: Whether to include 3D descriptors (requires conformers)
            
        Returns:
            List of dictionaries containing compound data and similarity scores
        """
        results = []
        
        # Parse query molecule
        query_mol = Chem.MolFromSmiles(query_smiles)
        if query_mol is None:
            raise ValueError(f"Invalid query SMILES: {query_smiles}")
        
        # Generate query fingerprint
        if fingerprint_type not in self.fingerprint_types:
            raise ValueError(f"Unsupported fingerprint type: {fingerprint_type}")
        
        try:
            query_fp = self.fingerprint_types[fingerprint_type](query_mol)
        except Exception as e:
            raise ValueError(f"Error generating query fingerprint: {e}")
        
        # Screen compound library
        valid_compounds = 0
        for i, smiles in enumerate(compound_library):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                
                # Apply property filters if requested
                if apply_filters and not self._apply_filters(mol, custom_filters):
                    continue
                
                # Generate fingerprint for library compound
                compound_fp = self.fingerprint_types[fingerprint_type](mol)
                
                # Calculate similarity
                similarity = self._calculate_similarity(query_fp, compound_fp, similarity_metric)
                
                if similarity >= similarity_threshold:
                    # Calculate properties
                    properties = self._calculate_properties(mol)
                    
                    result = {
                        'index': i,
                        'smiles': smiles,
                        'similarity': similarity,
                        'properties': properties
                    }
                    
                    # Add 3D descriptors if requested
                    if include_3d:
                        try:
                            # Generate conformer for 3D descriptors
                            mol_copy = Chem.AddHs(mol)
                            AllChem.EmbedMolecule(mol_copy, randomSeed=42)
                            AllChem.OptimizeMolecule(mol_copy)
                            
                            result['properties']['pmi1'] = rdMolDescriptors.PMI1(mol_copy)
                            result['properties']['pmi2'] = rdMolDescriptors.PMI2(mol_copy)
                            result['properties']['pmi3'] = rdMolDescriptors.PMI3(mol_copy)
                            result['properties']['spherocity'] = rdMolDescriptors.SpherocityIndex(mol_copy)
                        except Exception as e:
                            logger.warning(f"Error calculating 3D descriptors for compound {i}: {e}")
                    
                    results.append(result)
                    valid_compounds += 1
            
            except Exception as e:
                logger.warning(f"Error processing compound {i}: {e}")
                continue
        
        # Sort by similarity (descending) and limit results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        results = results[:max_results]
        
        logger.info(f"Screened {len(compound_library)} compounds, found {len(results)} hits above threshold {similarity_threshold}")
        
        return results
    
    def generate_pharmacophore_features(self, smiles: str) -> Dict[str, Any]:
        """
        Generate pharmacophore features for a molecule.
        
        Args:
            smiles: SMILES string of the molecule
            
        Returns:
            Dictionary containing pharmacophore features
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        try:
            # Add explicit hydrogens for better feature detection
            mol = Chem.AddHs(mol)
            
            # Generate 3D conformer
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.OptimizeMolecule(mol)
            
            # Define feature factory
            fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
            
            # Get features
            features = fdef.GetFeaturesForMol(mol)
            
            # Organize features by type
            feature_summary = {}
            for feat in features:
                feat_type = feat.GetType()
                if feat_type not in feature_summary:
                    feature_summary[feat_type] = 0
                feature_summary[feat_type] += 1
            
            return {
                'smiles': smiles,
                'num_features': len(features),
                'feature_types': feature_summary,
                'pharmacophore_features': [
                    {
                        'type': feat.GetType(),
                        'position': feat.GetPos(),
                        'atoms': list(feat.GetAtomIds())
                    } for feat in features
                ]
            }
            
        except Exception as e:
            logger.error(f"Error generating pharmacophore features: {e}")
            return {'error': str(e)}
    
    def batch_similarity_search(
        self,
        query_smiles: str,
        library_file: str,
        output_file: str,
        **kwargs
    ) -> str:
        """
        Perform batch similarity search from file input.
        
        Args:
            query_smiles: SMILES string of the reference ligand
            library_file: Path to file containing SMILES (one per line or CSV)
            output_file: Path to output CSV file
            **kwargs: Additional parameters for screen_compounds
            
        Returns:
            Status message
        """
        try:
            # Read compound library
            library_path = Path(library_file)
            if not library_path.exists():
                raise FileNotFoundError(f"Library file not found: {library_file}")
            
            if library_path.suffix.lower() == '.csv':
                df = pd.read_csv(library_file)
                if 'smiles' in df.columns:
                    compound_library = df['smiles'].tolist()
                else:
                    compound_library = df.iloc[:, 0].tolist()
            else:
                with open(library_file, 'r') as f:
                    compound_library = [line.strip() for line in f if line.strip()]
            
            # Perform screening
            results = self.screen_compounds(query_smiles, compound_library, **kwargs)
            
            # Convert to DataFrame and save
            if results:
                df_results = pd.DataFrame(results)
                # Flatten properties dict
                props_df = pd.json_normalize(df_results['properties'])
                df_final = pd.concat([df_results.drop('properties', axis=1), props_df], axis=1)
                df_final.to_csv(output_file, index=False)
                
                return f"Successfully screened {len(compound_library)} compounds. Found {len(results)} hits. Results saved to {output_file}"
            else:
                return f"No compounds found above similarity threshold. Screened {len(compound_library)} compounds."
                
        except Exception as e:
            error_msg = f"Error in batch similarity search: {e}"
            logger.error(error_msg)
            return error_msg


# Initialize global screener instance
screener = LigandBasedScreener()


@tool
def ligand_based_screening_tool(
    query_smiles: str,
    compound_library: Union[List[str], str],
    fingerprint_type: str = 'morgan',
    similarity_threshold: float = 0.7,
    max_results: int = 100,
    similarity_metric: str = 'tanimoto',
    apply_filters: bool = True,
    custom_filters: Optional[str] = None,
    include_3d: bool = False,
    output_file: Optional[str] = None
) -> str:
    """
    Perform ligand-based virtual screening using similarity searches and pharmacophore modeling.
    
    This tool searches for compounds similar to a reference ligand using molecular fingerprints
    and similarity calculations. It can filter compounds by drug-like properties and generate
    ranked lists of similar compounds.
    
    Args:
        query_smiles (str): SMILES string of the reference ligand to search for similar compounds
        compound_library (Union[List[str], str]): Either a list of SMILES strings or path to a file containing SMILES
        fingerprint_type (str): Type of molecular fingerprint to use. Options: 'morgan', 'rdkit', 'atom_pairs', 'topological_torsion', 'maccs'. Default: 'morgan'
        similarity_threshold (float): Minimum similarity score (0.0-1.0) for compounds to be included in results. Default: 0.7
        max_results (int): Maximum number of results to return, sorted by similarity score. Default: 100
        similarity_metric (str): Similarity metric to use. Options: 'tanimoto', 'dice', 'cosine'. Default: 'tanimoto'
        apply_filters (bool): Whether to apply drug-like property filters (Lipinski's rule, etc.). Default: True
        custom_filters (Optional[str]): JSON string of custom property filters as {"property": [min, max]}. Example: '{"mw": [200, 400], "logp": [-1, 3]}'
        include_3d (bool): Whether to include 3D molecular descriptors (requires conformer generation). Default: False
        output_file (Optional[str]): Path to save results as CSV file. If None, returns formatted string results
    
    Returns:
        str: Formatted results containing similar compounds with their similarity scores and properties,
             or status message if output_file is specified
    
    Examples:
        # Basic similarity search
        result = ligand_based_screening_tool(
            query_smiles="CCO",
            compound_library=["CCO", "CCCO", "CCCCO", "c1ccccc1O"],
            similarity_threshold=0.5
        )
        
        # Advanced search with custom filters
        result = ligand_based_screening_tool(
            query_smiles="CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
            compound_library="compounds.smi",
            fingerprint_type="morgan",
            custom_filters='{"mw": [150, 300], "logp": [0, 4]}',
            output_file="aspirin_analogs.csv"
        )
    """
    try:
        # Input validation
        if not query_smiles or not isinstance(query_smiles, str):
            return "Error: query_smiles must be a non-empty string"
        
        # Parse custom filters if provided
        parsed_filters = None
        if custom_filters:
            try:
                import json
                parsed_filters = json.loads(custom_filters)
                # Convert lists to tuples for filter format
                parsed_filters = {k: tuple(v) if isinstance(v, list) else v for k, v in parsed_filters.items()}
            except Exception as e:
                return f"Error: Invalid custom_filters JSON format: {e}"
        
        # Handle compound library input
        if isinstance(compound_library, str):
            # Check if it's a file path
            if os.path.exists(compound_library):
                if output_file:
                    # Use batch processing for file input with output file
                    return screener.batch_similarity_search(
                        query_smiles=query_smiles,
                        library_file=compound_library,
                        output_file=output_file,
                        fingerprint_type=fingerprint_type,
                        similarity_threshold=similarity_threshold,
                        max_results=max_results,
                        similarity_metric=similarity_metric,
                        apply_filters=apply_filters,
                        custom_filters=parsed_filters,
                        include_3d=include_3d
                    )
                else:
                    # Read file and process normally
                    try:
                        if compound_library.endswith('.csv'):
                            import pandas as pd
                            df = pd.read_csv(compound_library)
                            if 'smiles' in df.columns:
                                compound_list = df['smiles'].tolist()
                            else:
                                compound_list = df.iloc[:, 0].tolist()
                        else:
                            with open(compound_library, 'r') as f:
                                compound_list = [line.strip() for line in f if line.strip()]
                    except Exception as e:
                        return f"Error reading compound library file: {e}"
            else:
                # Treat as single SMILES string
                compound_list = [compound_library]
        else:
            compound_list = compound_library
        
        # Perform screening
        results = screener.screen_compounds(
            query_smiles=query_smiles,
            compound_library=compound_list,
            fingerprint_type=fingerprint_type,
            similarity_threshold=similarity_threshold,
            max_results=max_results,
            similarity_metric=similarity_metric,
            apply_filters=apply_filters,
            custom_filters=parsed_filters,
            include_3d=include_3d
        )
        
        # Handle output
        if output_file:
            try:
                import pandas as pd
                if results:
                    df_results = pd.DataFrame(results)
                    # Flatten properties dict
                    props_df = pd.json_normalize(df_results['properties'])
                    df_final = pd.concat([df_results.drop('properties', axis=1), props_df], axis=1)
                    df_final.to_csv(output_file, index=False)
                    return f"Successfully screened {len(compound_list)} compounds. Found {len(results)} hits above similarity threshold {similarity_threshold}. Results saved to {output_file}"
                else:
                    return f"No compounds found above similarity threshold {similarity_threshold}. Screened {len(compound_list)} compounds."
            except Exception as e:
                return f"Error saving results to file: {e}"
        else:
            # Format results as string
            if not results:
                return f"No compounds found above similarity threshold {similarity_threshold}. Screened {len(compound_list)} compounds."
            
            output_lines = [
                f"Ligand-based Virtual Screening Results",
                f"Query SMILES: {query_smiles}",
                f"Fingerprint: {fingerprint_type}",
                f"Similarity metric: {similarity_metric}",
                f"Threshold: {similarity_threshold}",
                f"Compounds screened: {len(compound_list)}",
                f"Hits found: {len(results)}",
                "",
                "Top Results:",
                "=" * 80
            ]
            
            for i, result in enumerate(results[:20], 1):  # Show top 20
                props = result['properties']
                output_lines.extend([
                    f"Rank {i}: Similarity = {result['similarity']:.3f}",
                    f"  SMILES: {result['smiles']}",
                    f"  MW: {props.get('mw', 'N/A'):.1f}, LogP: {props.get('logp', 'N/A'):.2f}, " +
                    f"HBD: {props.get('hbd', 'N/A')}, HBA: {props.get('hba', 'N/A')}",
                    f"  Rotatable bonds: {props.get('rotatable_bonds', 'N/A')}, TPSA: {props.get('tpsa', 'N/A'):.1f}",
                    ""
                ])
            
            if len(results) > 20:
                output_lines.append(f"... and {len(results) - 20} more compounds")
            
            return "\n".join(output_lines)
            
    except Exception as e:
        error_msg = f"Error in ligand-based virtual screening: {str(e)}"
        logger.error(error_msg)
        return error_msg


if __name__ == "__main__":
    # Test the tool
    test_query = "CCO"  # Ethanol
    test_library = ["CCO", "CCCO", "CCCCO", "c1ccccc1O", "CC(C)O", "CCCC"]
    
    print("Testing ligand_based_screening_tool...")
    result = ligand_based_screening_tool(
        query_smiles=test_query,
        compound_library=test_library,
        similarity_threshold=0.3,
        max_results=10
    )
    print(result)
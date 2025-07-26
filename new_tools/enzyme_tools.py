import os
import pandas as pd
import subprocess
from smolagents import CodeAgent
from smolagents import tool
from smolagents import LiteLLMModel
from typing import Dict, List, Any, Optional, Tuple


@tool
def run_mmseqs_cluster(
        target_db: str,
        result_file: str,
        tmp_dir: str,
        c: float = 0.8,
        min_seq_id: float = 0.0,
        threads: int = 32
) -> Dict[str, Any]:
    """
    Using MMseqs to clusters the entries of a FASTA/FASTQ file

    Args:
        target_db: Path to the FASTA file to be clustered.
        result_file: Path to store the search results.
        tmp_dir: A temporary directory for MMseqs to use.
        c: List matches above this fraction of aligned (covered) residues (default 0.8).
        min_seq_id:  List matches above this sequence identity for clustering, ranging 0.0-1.0 (default 0.0).
        threads: Number of CPU-cores used (default 32).
    Returns:
        Dictionary containing the path to the searching results.
    """
    try:
        if not os.path.exists(target_db):
            raise FileNotFoundError(f"The fasta file not found at {target_db}")

        os.makedirs(tmp_dir, exist_ok=True)

        # Construct the MMseqs command
        command_parts = []
        command_parts.extend(["conda", "run", "-n", "mmseqs"])

        command_parts.extend([
            "mmseqs", "easy-cluster", target_db, result_file, tmp_dir,
            "-c", str(c),
            "--min-seq-id", str(min_seq_id),
            "--threads", str(threads),
        ])

        command = " ".join(command_parts)
        # os.system(command)
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)

        return {
            "status": "success",
            "results_path": result_file
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
        }


@tool
def run_mmseqs_search(
        query_fasta: str,
        target_db: str,
        result_file: str,
        tmp_dir: str,
        s: float = 5.7,
        max_seqs: int = 300,
        min_seq_id: float = 0.0,
        c: float = 0.0,
        threads: int = 32
) -> Dict[str, Any]:
    """
    Using MMseqs 'easy-search' to search directly with a FASTA/FASTQ files against either another FASTA/FASTQ file

    Args:
        query_fasta: Path to the query FASTA file.
        target_db: Path to the target FASTA file.
        result_file: Path to store the search results.
        tmp_dir: A temporary directory for MMseqs to use.
        s: Sensitivity setting. Higher is more sensitive but slower (default: 5.7).
        max_seqs: Maximum results per query sequence allowed to pass the prefilter (default 300).
        min_seq_id: List matches above this sequence identity, ranging 0.0-1.0 (default 0.0).
        c:  List matches above this fraction of aligned (covered) residues (default 0.0).
        threads: Number of CPU-cores used (default 32).
    Returns:
        Dictionary containing the path to the searching results.
    """
    try:
        if not os.path.exists(query_fasta):
            raise FileNotFoundError(f"The query fasta file not found at {query_fasta}")

        if not os.path.exists(target_db):
            raise FileNotFoundError(f"The target fasta file not found at {target_db}")

        os.makedirs(tmp_dir, exist_ok=True)

        # Construct the MMseqs command
        command_parts = []
        command_parts.extend(["conda", "run", "-n", "mmseqs"])

        command_parts.extend([
            "mmseqs", "easy-search", query_fasta, target_db, result_file, tmp_dir,
            "-s", str(s),
            "--max-seqs", str(max_seqs),
            "--min-seq-id", str(min_seq_id),
            "-c", str(c),
            "--threads", str(threads),
        ])

        command = " ".join(command_parts)
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True)

        return {
            "status": "success",
            "results_path": result_file
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
        }


@tool
def parse_mmseqs_search_results(
        result_file: str,
        target_database: str
) -> Dict[str, Any]:
    """
    Parses a standard MMseqs2 tabular result file (.m8 format).

    The default output format has 12 columns:
    1.  query:      Query identifier
    2.  target:     Target identifier
    3.  pident:     Percentage of identical matches
    4.  alnlen:     Alignment length
    5.  mismatch:   Number of mismatches
    6.  gapopen:    Number of gap openings
    7.  qstart:     Start of alignment in query
    8.  qend:       End of alignment in query
    9.  tstart:     Start of alignment in target
    10. tend:       End of alignment in target
    11. evalue:     Expectation value (E-value)
    12. bits:       Bit score

    Args:
        result_file: The path to the MMseqs2/Foldseek result file.
        target_database: The FASTA sequence database for the searching targets.
    Returns:
        Dictionary containing the parse status and the hits targets,
            amino acid sequence of each target, percentage of identical matches of each target,
             e_value of each target.
    """
    # Define the standard column names for the tabular output
    columns = [
        "query", "target", "pident", "alnlen", "mismatch", "gapopen",
        "qstart", "qend", "tstart", "tend", "evalue", "bits"
    ]

    parsed_data = {}
    target_names = set()
    try:
        if not os.path.exists(result_file):
            raise FileNotFoundError(f"Result file not found at: {result_file}")

        if not os.path.exists(target_database):
            raise FileNotFoundError(f"Target database not found at: {target_database}")

        with open(result_file, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                fields = line.split('\t')

                if len(fields) != len(columns):
                    raise ValueError(f"Only support output file format with columns {' '.join(columns)}")

                hit_data = dict(zip(columns, fields))

                target_names.add(hit_data["target"])
                hit_data["pident"] = float(hit_data["pident"])
                hit_data["evalue"] = float(hit_data["evalue"])

                query = hit_data["query"]
                if query in parsed_data.keys():
                    parsed_data[query]["targets"].append(hit_data["target"])
                    parsed_data[query]["pidents"].append(hit_data["pident"])
                    parsed_data[query]["evalues"].append(hit_data["evalue"])
                else:
                    parsed_data[query] = {
                        "targets": [hit_data["target"]],
                        "pidents": [hit_data["pident"]],
                        "evalues": [hit_data["evalue"]]
                    }

        name2seq = {}
        with open(target_database, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines[::2]):
            line = line.strip()
            if line.startswith(">") and line[1:] in target_names:
                name2seq[line[1:]] = lines[i + 1].strip()
        seq_list = []
        for query, info in parsed_data.items():
            for target in info["targets"]:
                seq_list.append(name2seq[target])
            parsed_data[query]["sequence"] = seq_list

        return {
            "status": "success",
            "data": parsed_data,
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e)
        }


# @tool
def create_foldseek_database(
        input_path: str,
        db_path: str,
        threads: int = 32,
) -> Dict[str, Any]:
    """
    Creates a Foldseek database from a directory of PDB or mmCIF files.

    Args:
        input_path: The path to the input directory containing PDB/mmCIF structure files.
        db_path: The path and prefix for the output database files. Foldseek will create
                       multiple files with this prefix.
        threads: Number of CPU-cores used (default 32).

    Returns:
       Dictionary containing the status and results of the operation.
    """
    try:
        if not os.path.isdir(input_path):
            raise FileNotFoundError(f"The input directory was not found at: {input_path}")

        output_dir = os.path.dirname(db_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        command_parts = []
        command_parts.extend(["conda", "run", "-n", "foldseek"])

        command_parts.extend([
            "foldseek", "createdb", input_path, db_path,
            "--threads", str(threads)
        ])

        command = " ".join(command_parts)
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )

        return {
            "status": "success",
            "db_path": db_path,
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e)
        }


# @tool
def run_foldseek_search(
        query_db: str,
        target_db: str,
        result_file: str,
        tmp_dir: str,
        s: float = 9.5,
        e_value: float = 0.001,
        max_seqs: int = 1000,
        alignment_type: int = 3,
        threads: int = 8,
) -> Dict[str, Any]:
    """
    Uses Foldseek 'easy-search' to search a query database of protein structures (PDB/CIF)
    against a target database.

    Args:
        query_db: Path to the query structures (directory of PDB/CIF files or a Foldseek DB).
        target_db: Path to the target structures (directory of PDB/CIF files or a Foldseek DB).
        result_file: Path to store the search results in alignment format.
        tmp_dir: A temporary directory for Foldseek to use. .
        s : Sensitivity setting. Higher is more sensitive but slower (default 9.5).
        e_value: E-value threshold for reporting hits (default 0.001).
        max_seqs: Maximum results per query to pass the prefilter (default 1000).
        alignment_type: How to compute the alignment: 0: 3di alignment, 1: TM alignment, 2: 3Di+AA (default 2)
        threads : Number of CPU-cores used (default 32).

    Returns:
        Dictionary containing the path to the searching results.
    """
    try:
        if not os.path.exists(query_db):
            raise FileNotFoundError(f"Query structures not found at: {query_db}")
        if not os.path.exists(target_db):
            raise FileNotFoundError(f"Target structures not found at: {target_db}")

        os.makedirs(tmp_dir, exist_ok=True)

        command_parts = []
        command_parts.extend(["conda", "run", "-n", "foldseek"])

        command_parts.extend([
            "foldseek", "easy-search", query_db, target_db, result_file, tmp_dir,
            "--threads", str(threads),
            "-s", str(s),
            "-e", str(e_value),
            "--max-seqs", str(max_seqs),
            "--alignment-type", str(alignment_type)
        ])

        command = " ".join(command_parts)
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )

        return {
            "status": "success",
            "results_path": os.path.abspath(result_file),
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e)
        }


@tool
def run_ephod_prediction(sequence_path: str, save_dir: str, csv_name: str) -> Dict[str, Any]:
    """
    Use a deep-learning model Ephod to predict the optimum pH from the sequence of enzymes.

     Args:
        sequence_path: Path to the sequence of enzymes.
        save_dir: Directory to store the results.
        csv_name: Name of the output csv file.
    Returns:
        Dictionary containing the path to the searching results and the optimum pH.
    """
    try:
        if not os.path.exists(sequence_path):
            raise FileNotFoundError(f"Sequence file not found at: {sequence_path}")

        os.makedirs(save_dir, exist_ok=True)

        command_parts = []
        command_parts.extend(["conda", "run", "-n", "ephod"])

        command_parts.extend([
            "ephod",
            "--fasta_path", sequence_path,
            "--save_dir", save_dir,
            "--csv_name", csv_name,
            "--verbose", "1",
            "--save_attention_weights", "0",
            "--save_embeddings", "0"
        ])

        command = " ".join(command_parts)

        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )

        # Parse results file
        name2seq = {}
        parsed_results = []
        with open(sequence_path) as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith(">"):
                name = line[1:].strip().split()[0]
            else:
                name2seq[name] = line.strip()

        csv_path = os.path.join(save_dir, csv_name)
        with open(csv_path) as f:
            lines = f.readlines()
        for line in lines[1:]:
            items = line.split(",")
            prot_name, optpH = items[0], items[3]
            seq = name2seq[prot_name]
            parsed_results.append({
                "prot_name": prot_name,
                "seq": seq,
                "optpH": optpH,
            })

        return {
            "status": "success",
            "results_path": os.path.abspath(csv_path),
            "parsed_results": parsed_results
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e)
        }


@tool
def run_boltz_protein_structure_prediction(sequence_path: str, output_dir: Optional[str]) -> Dict[str, Any]:
    """
    Use Boltz2 to predict the 3D structure of protein sequence.

     Args:
        sequence_path: Path to the protein sequence.
        output_dir: Directory to store the results.
    Returns:
        Dictionary containing the path to the mmCIF file of the predicted protein structure.
    """
    try:
        if not os.path.exists(sequence_path):
            raise FileNotFoundError(f"Sequence file not found at: {sequence_path}")

        parent_dir = os.path.dirname(sequence_path)
        input_dir = os.path.join(parent_dir, "boltz_input")
        if not output_dir:
            output_dir = os.path.join(parent_dir, "boltz_output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        with open(sequence_path) as f:
            lines = f.readlines()
        name_list = []
        for i in range(len(lines))[::2]:
            name = lines[i][1:].split()[0]
            name_list.append(name)
            out = f">A|protein\n"
            out += lines[i + 1]
            with open(os.path.join(input_dir, name + ".fasta"), "w") as fw:
                fw.write(out)

        command_parts = []
        command_parts.extend(["conda", "run", "-n", "boltz2"])

        command_parts.extend([
            "boltz",
            "predict", input_dir,
            "--out_dir", output_dir,
            "--use_msa_server",
            "--msa_server_username", "myuser",
            "--msa_server_password", "mypassword",
        ])

        command = " ".join(command_parts)

        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )

        preds_dir = os.path.join(output_dir, "boltz_results_input", "predictions")
        results_list = []
        for name in name_list:
            structure_path = os.path.join(preds_dir, name, f"{name}_model_0.cif")
            if not os.path.exists(structure_path):
                raise ValueError("Prediction failed due to inner error.")
            results_list.append({
                "name": name,
                "structure_path": structure_path
            })
        return {
            "status": "success",
            "results_path": results_list,
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e)
        }


@tool
def run_clean_ec_prediction(sequence_path: str) -> Dict[str, Any]:
    """
    Use CLEAN to predict EC number of enzyme(protein) sequence.

     Args:
        sequence_path: Path to the protein sequence.
    Returns:
        Dictionary containing the EC number and distance to the cluster center of the protein.
    """
    CLEAN_DIR = "/home/ubuntu/file2/CLEAN/app/"
    try:
        if not os.path.exists(sequence_path):
            raise FileNotFoundError(f"Sequence file not found at: {sequence_path}")

        parent_dir = os.path.dirname(sequence_path)
        output_dir = os.path.join(parent_dir, "clean_output")
        # input_dir = os.path.join(parent_dir, "input")

        # os.makedirs(input_dir, exist_ok=True)
        # os.makedirs(output_dir, exist_ok=True)

        cur_workspace = os.getcwd()
        os.chdir(CLEAN_DIR)
        command_parts = []
        command_parts.extend(["conda", "run", "-n", "clean"])

        command_parts.extend([
            "python", "CLEAN_infer_fasta.py",
            "--fasta_data", sequence_path
        ])

        command = " ".join(command_parts)
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        os.chdir(cur_workspace)

        csv_path = os.path.join(output_dir, "output_maxsep.csv")
        results_list = []
        with open(csv_path) as f:
            lines = f.readlines()
        for line in lines:
            name, ec = line.strip().split(",")
            ec, distance = ec.split("/")
            results_list.append({
                "name": name,
                "EC number": ec[3:],
                "distance": distance
            })
        return {
            "status": "success",
            "results": results_list,
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e)
        }


@tool
def run_catapro_prediction(sequence_path: str, smiles_path: str) -> Dict[str, Any]:
    """
    Uses CataPro to predict enzyme kinetic parameters including turnover number (kcat), Michaelis constant (Km),
     and catalytic efficiency (kcat/Km) given the Enzyme and substrate information.

     Args:
        sequence_path: Path to the protein sequence.
        smiles_path: Path to smiles of the substrate.
    Returns:
        Dictionary containing kcat, Km, kcat/Km.
    """
    CATAPRO_DIR = "/home/ubuntu/file2/CataPro/"
    try:
        if not os.path.exists(sequence_path):
            raise FileNotFoundError(f"Sequence file not found at: {sequence_path}")

        if not os.path.exists(smiles_path):
            raise FileNotFoundError(f"Molecule smiles file not found at: {smiles_path}")

        parent_dir = os.path.dirname(sequence_path)
        input_dir = os.path.join(parent_dir, "catapro_input")
        output_dir = os.path.join(parent_dir, "catapro_output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Construct csv
        with open(sequence_path) as f:
            fasta_lines = f.readlines()
        with open(smiles_path) as f:
            smiles_lines = f.readlines()
        csv_data = ",Enzyme_id,type,sequence,smiles\n"
        for i in range(len(smiles_lines)):
            name = fasta_lines[2 * i][1:].strip().split()[0]
            seq = fasta_lines[2 * i + 1].strip()
            smile = smiles_lines[i]
            csv_data += f"{i},{name},wild,{seq},{smile}\n"
        csv_path = os.path.join(input_dir, "input.csv")
        with open(csv_path, "w") as f:
            f.write(csv_data)

        # input_dir = os.path.join(parent_dir, "input")

        cur_workspace = os.getcwd()
        os.chdir(CATAPRO_DIR)
        output_path = os.path.join(output_dir, "output.csv")
        command_parts = []
        command_parts.extend(["conda", "run", "-n", "catapro"])
        command_parts.extend([
            "python", "inference/predict.py",
            "-inp_fpath", csv_path,
            "-model_dpath", "models",
            "-batch_size", "64",
            "-out_fpath", output_path
        ])

        command = " ".join(command_parts)
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        os.chdir(cur_workspace)

        results_list = []
        with open(output_path) as f:
            lines = f.readlines()
        for line in lines[1:]:
            _, name, smiles, kcat, km, kcat_km = line.strip().split(",")
            results_list.append({
                "name": name,
                "smiles": smiles,
                "kcat": kcat,
                "km": km,
                "kcat/km": kcat_km
            })
        return {
            "status": "success",
            "results": results_list,
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e)
        }


@tool
def run_prime_ogt_prediction(sequence_path: str) -> Dict[str, Any]:
    """
    Uses Prime to predict the optimal growth temperature of the enzyme.

     Args:
        sequence_path: Path to the enzyme(protein) sequence.
    Returns:
        Dictionary containing the optimal growth temperature of the enzyme.
    """
    PRIME_DIR = "/home/ubuntu/file2/Pro-Prime/"
    try:
        if not os.path.exists(sequence_path):
            raise FileNotFoundError(f"Sequence file not found at: {sequence_path}")

        parent_dir = os.path.dirname(sequence_path)
        output_dir = os.path.join(parent_dir, "prime_output")
        os.makedirs(output_dir, exist_ok=True)

        cur_workspace = os.getcwd()
        os.chdir(PRIME_DIR)
        command_parts = []
        command_parts.extend(["conda", "run", "-n", "prime"])
        command_parts.extend([
            "python", "predict_OGT.py",
            "--fasta_path", sequence_path,
            "--save_dir", output_dir,
            "--csv_name", "prediction.csv"
        ])

        command = " ".join(command_parts)
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        os.chdir(cur_workspace)

        csv_path = os.path.join(output_dir, "prediction.csv")
        results_list = []
        with open(csv_path) as f:
            lines = f.readlines()
        for line in lines[1:]:
            name, ogt = line.strip().split(",")
            results_list.append({
                "name": name,
                "optimal growth temperature": ogt
            })
        return {
            "status": "success",
            "results": results_list,
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e)
        }


@tool
def run_chroma_redesign(structure_path: str) -> Dict[str, Any]:
    """
    Uses Chroma to redesign a known protein or enzyme.

     Args:
        structure_path: Path to the protein structure in PDB or MMCIF format.
    Returns:
        Dictionary containing the path to the redesigned protein structure.
    """
    CHROMA_DIR = "/home/ubuntu/file2/Chroma/"
    try:
        if not os.path.exists(structure_path):
            raise FileNotFoundError(f"Structure file not found at: {structure_path}")

        parent_dir = os.path.dirname(structure_path)
        base_name = os.path.basename(structure_path)
        output_dir = os.path.join(parent_dir, "chroma_output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, base_name)

        cur_workspace = os.getcwd()
        os.chdir(CHROMA_DIR)
        command_parts = []
        command_parts.extend(["conda", "run", "-n", "chroma"])
        command_parts.extend([
            "python", "gen.py",
            "--protein_path", structure_path,
            "--output_path", output_path
        ])

        command = " ".join(command_parts)
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        os.chdir(cur_workspace)

        return {
            "status": "success",
            "results_path": output_path,
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e)
        }


@tool
def run_iqtree_reconstruct_phylogenetic_trees(msa_path: str) -> Dict[str, Any]:
    """
    Uses IQ-TREE to reconstruct phylogenetic trees.

     Args:
        msa_path: Path to the multiple sequence alignment file of the desired protein.
    Returns:
        Dictionary containing the path to the constructed phylogenetic trees.
    """
    try:
        if not os.path.exists(msa_path):
            raise FileNotFoundError(f"MSA file not found at: {msa_path}")

        command_parts = []
        command_parts.extend(["conda", "run", "-n", "iqtree"])
        command_parts.extend([
            "iqtree",
            "-s", msa_path,
        ])

        command = " ".join(command_parts)
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )

        return {
            "status": "success",
            "results_path": os.path.join(msa_path + ".iqtree"),
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e)
        }


@tool
def run_ligandmpnn_redesign(structure_path: str) -> Dict[str, Any]:
    """
    Uses LigandMPNN to redesign a known protein or enzyme.

     Args:
        structure_path: Path to the protein structure in PDB or MMCIF format.
    Returns:
        Dictionary containing the path to the redesigned protein structure.
    """
    LIGANDMPNN_DIR = "/home/ubuntu/file2/LigandMPNN/"
    try:
        if not os.path.exists(structure_path):
            raise FileNotFoundError(f"Structure file not found at: {structure_path}")

        parent_dir = os.path.dirname(structure_path)
        base_name = os.path.basename(structure_path)
        output_dir = os.path.join(parent_dir, "ligandmpnn_output")
        os.makedirs(output_dir, exist_ok=True)

        cur_workspace = os.getcwd()
        os.chdir(LIGANDMPNN_DIR)
        command_parts = []
        command_parts.extend(["conda", "run", "-n", "ligandmpnn_env"])
        command_parts.extend([
            "python", "run.py",
            "--seed", "111",
            "--pdb_path", structure_path,
            "--out_folder", output_dir,
        ])

        command = " ".join(command_parts)
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        # os.system(command)
        os.chdir(cur_workspace)

        structure_path = os.path.join(output_dir, "backbones", base_name + "_1.pdb")
        seq_path = os.path.join(output_dir, "seqs", base_name + ".fa")
        with open(seq_path) as f:
            seq = f.readlines()[1]

        return {
            "status": "success",
            "structure_path": structure_path,
            "seq_path": seq_path,
            "sequence": seq
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e)
        }

# if __name__ == "__main__":
#     input_fasta = "/home/ubuntu/agents/agent_outputs/enzyme_tool_debug/query.fasta"
#     os.chdir("/home/ubuntu/file2/CLEAN/app/")
#     os.system(
#         "conda run -n clean python CLEAN_infer_fasta.py --fasta_data '/home/ubuntu/agents/agent_outputs/enzyme_tool_debug/query.fasta'")

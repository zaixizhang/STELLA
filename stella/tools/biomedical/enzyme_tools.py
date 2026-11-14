import os
import re
import shutil
import json
import pandas as pd
import subprocess
import requests
from smolagents import tool
from typing import Dict, List, Any, Optional, Tuple
from Bio import SeqIO
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

aa_dict = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y"
}


def check_conda_env(env_name):
    try:
        result = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True, check=True)
        return re.search(rf'^{re.escape(env_name)}\s+', result.stdout, re.MULTILINE) is not None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_cur_time():
    import datetime
    now = datetime.datetime.now()
    formatted_timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    return formatted_timestamp


def read_seq_from_fasta(fasta_path):
    with open(fasta_path) as f:
        lines = f.readlines()
    seq_list = []
    name, seq = "", ""
    for line in lines:
        line = line.strip()
        if line.startswith(">"):
            if name:
                seq_list.append((name, seq))
            name = line[1:]
            seq = ""
        else:
            seq += line
    seq_list.append((name, seq))
    return seq_list


@tool
def extract_seq_from_structure(structure_path: str) -> Dict[str, Any]:
    """
        Extract the protein sequence from the protein structure in PDB/MMCIF format.

        Args:
            structure_path: The path to the protein structure.
        Returns:
            Dictionary containing the path to the protein fasta.
        """
    try:
        if not os.path.exists(structure_path):
            raise FileNotFoundError(f"{structure_path} not exsited.")

        parent_dir = os.path.dirname(structure_path)
        base_name = os.path.basename(structure_path)
        seq_dir = os.path.join(parent_dir, f"seq_{get_cur_time()}")
        os.makedirs(seq_dir, exist_ok=True)
        seq_path = os.path.join(seq_dir, base_name.split(".")[0] + ".fasta")

        file_extension = os.path.splitext(structure_path)[1].lower()
        if file_extension == '.cif' or file_extension == '.mmcif':
            parser = MMCIFParser(QUIET=True)
            file_type = "mmCIF"
        elif file_extension == '.pdb':
            parser = PDBParser(QUIET=True)
            file_type = "PDB"
        else:
            raise ValueError("Unexpected file type.")

        structure = parser.get_structure('structure', structure_path)

        fasta_records = []
        for model in structure:
            for chain in model:
                sequence_3_letter = []
                for residue in chain:
                    if residue.id[0] == ' ':
                        sequence_3_letter.append(residue.get_resname())

                if sequence_3_letter:
                    sequence_1_letter = "".join([aa_dict[res] for res in sequence_3_letter])
                    record_id = f"{structure.id}|{model.id}|{chain.id}"
                    record = SeqRecord(
                        Seq(sequence_1_letter),
                        id=record_id,
                        description=f"Chain {chain.id} from {structure_path}"
                    )
                    fasta_records.append(record)

        if fasta_records:
            SeqIO.write(fasta_records, seq_path, "fasta")
        else:
            raise ValueError("No valid residue found.")

        return {
            "status": "success",
            "sequence_path": seq_path
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
        }


@tool
def extract_pocket(complex_path: str, distance_cutoff: float = 5.0) -> Dict[str, Any]:
    """
        Extract the pocket given a protein-ligand complex.

        Args:
            complex_path: The path to the complex structure.
            distance_cutoff: Binding pocket distance cutoff.(default 5.0 Angstrom)
        Returns:
            Dictionary containing the residue ids of the pocket
            and the path to the protein, ligand and pocket structure.
        """

    class ProteinSelect(Select):
        def accept_residue(self, residue):
            return residue.id[0] == ' '

    class LigandSelect(Select):
        def __init__(self, ligand_id):
            self.ligand_id = ligand_id

        def accept_residue(self, residue):
            return residue.id[0].startswith('H_') and residue.get_resname() == self.ligand_id

    # TODO
    LIGAND_ID = 'LIG1'

    parent_dir = os.path.dirname(complex_path)
    pocket_dir = os.path.join(parent_dir, f"pocket_{get_cur_time()}")
    os.makedirs(pocket_dir, exist_ok=True)
    protein_path = os.path.join(pocket_dir, 'protein.pdb')
    ligand_path = os.path.join(pocket_dir, 'ligand.pdb')
    pocket_path = os.path.join(pocket_dir, 'pocket.pdb')

    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('protein_complex', complex_path)

        io = PDBIO()
        io.set_structure(structure)
        io.save(protein_path, ProteinSelect())

        io.set_structure(structure)
        io.save(ligand_path, LigandSelect(LIGAND_ID))

        ligand_atoms = []
        for residue in structure.get_residues():
            if residue.id[0].startswith('H_') and residue.get_resname() == LIGAND_ID:
                for atom in residue.get_atoms():
                    ligand_atoms.append(atom)

        if not ligand_atoms:
            raise ValueError(f"No ligand found in with ligand id {LIGAND_ID}")

        pocket_residues = set()

        res_ids = []
        for residue in structure.get_residues():
            if residue.id[0] == ' ':
                for protein_atom in residue.get_atoms():
                    for ligand_atom in ligand_atoms:
                        distance = protein_atom - ligand_atom
                        if distance < distance_cutoff:
                            res_ids.append(residue.get_id()[1])
                            pocket_residues.add(residue)
                            break
                    if residue in pocket_residues:
                        break

        class PocketSelect(Select):
            def __init__(self, residues):
                self.residues_to_keep = {res.get_full_id() for res in residues}

            def accept_residue(self, residue):
                return residue.get_full_id() in self.residues_to_keep

        io.set_structure(structure)
        io.save(pocket_path, PocketSelect(list(pocket_residues)))
        return {
            "status": "success",
            "residue_ids": res_ids,
            "protein_path": protein_path,
            "ligand_path": ligand_path,
            "pokcet_path": pocket_path
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
        }


@tool
def download_uniprot_seq(uniprot_id: str,
                         save_dir: str = "/home/ubuntu/agents/agent_outputs/enzyme_tool_debug") -> Dict[str, Any]:
    """
    Download the protein sequence with UniProt ID in FASTA format.

    Args:
        uniprot_id: The UniProt ID of the protein.
        save_dir: The directory to save the FASTA file (Optional).
    Returns:
        Dictionary containing the path to the FASTA file.
    """
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    save_path = os.path.join(save_dir, f"{uniprot_id}.fasta")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, 'w') as f:
            f.write(response.text)
        return {
            "status": "success",
            "results_path": save_path,
        }
    except requests.exceptions.RequestException as e:
        return {
            "status": "error",
            "error_message": str(e),
        }


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
    ENV_NAME = "mmseqs"
    try:
        if not os.path.exists(target_db):
            raise FileNotFoundError(f"The fasta file not found at {target_db}.")
        if not check_conda_env(ENV_NAME):
            raise OSError(f"The conda environment {ENV_NAME} not fould, please install {ENV_NAME} first!")

        os.makedirs(tmp_dir, exist_ok=True)

        # Construct the MMseqs command
        command_parts = []
        command_parts.extend(["conda", "run", "-n", ENV_NAME])

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
    ENV_NAME = "mmseqs"
    try:
        if not os.path.exists(query_fasta):
            raise FileNotFoundError(f"The query fasta file not found at {query_fasta}")

        if not os.path.exists(target_db):
            raise FileNotFoundError(f"The target fasta file not found at {target_db}")

        if not check_conda_env(ENV_NAME):
            raise OSError(f"The conda environment {ENV_NAME} not fould, please install {ENV_NAME} first!")

        os.makedirs(tmp_dir, exist_ok=True)

        # Construct the MMseqs command
        command_parts = []
        command_parts.extend(["conda", "run", "-n", ENV_NAME])

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
    ENV_NAME = "foldseek"
    try:
        if not os.path.isdir(input_path):
            raise FileNotFoundError(f"The input directory was not found at: {input_path}")

        if not check_conda_env(ENV_NAME):
            raise OSError(f"The conda environment {ENV_NAME} not fould, please install {ENV_NAME} first!")

        output_dir = os.path.dirname(db_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        command_parts = []
        command_parts.extend(["conda", "run", "-n", ENV_NAME])

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
    ENV_NAME = "foldseek"
    try:
        if not os.path.exists(query_db):
            raise FileNotFoundError(f"Query structures not found at: {query_db}")
        if not os.path.exists(target_db):
            raise FileNotFoundError(f"Target structures not found at: {target_db}")

        if not check_conda_env(ENV_NAME):
            raise OSError(f"The conda environment {ENV_NAME} not fould, please install {ENV_NAME} first!")

        os.makedirs(tmp_dir, exist_ok=True)

        command_parts = []
        command_parts.extend(["conda", "run", "-n", ENV_NAME])

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
    ENV_NAME = "ephod"
    try:
        if not os.path.exists(sequence_path):
            raise FileNotFoundError(f"Sequence file not found at: {sequence_path}")

        if not check_conda_env(ENV_NAME):
            raise OSError(f"The conda environment {ENV_NAME} not fould, please install {ENV_NAME} first!")

        os.makedirs(save_dir, exist_ok=True)

        command_parts = []
        command_parts.extend(["conda", "run", "-n", ENV_NAME])

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
def run_boltz_protein_structure_prediction(sequence_path: str) -> Dict[str, Any]:
    """
    Use Boltz2 to predict the 3D structure of protein sequence.

     Args:
        sequence_path: Path to the protein sequence.
    Returns:
        Dictionary containing the path to the mmCIF file of the predicted protein structure.
    """
    ENV_NAME = "boltz2"
    try:
        if not os.path.exists(sequence_path):
            raise FileNotFoundError(f"Sequence file not found at: {sequence_path}")

        if not check_conda_env(ENV_NAME):
            raise OSError(f"The conda environment {ENV_NAME} not fould, please install {ENV_NAME} first!")

        parent_dir = os.path.dirname(sequence_path)
        input_dir = os.path.join(parent_dir, "boltz_input")
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
        command_parts.extend(["conda", "run", "-n", ENV_NAME])

        command_parts.extend([
            "boltz",
            "predict", input_dir,
            "--out_dir", output_dir,
            "--use_msa_server",
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
def run_boltz_complex_structure_prediction(sequence_path: str, smiles_path: str) -> Dict[str, Any]:
    """
    Use Boltz2 to predict the 3D complex structure given a protein sequence and the smiles of ligand.

     Args:
        sequence_path: Path to the protein sequence.
        smiles_path: Path to smiles of the ligand.
    Returns:
        Dictionary containing the path to the mmCIF file of the predicted complex structure.
    """
    ENV_NAME = "boltz2"
    try:
        if not os.path.exists(sequence_path):
            raise FileNotFoundError(f"Sequence file not found at: {sequence_path}")

        if not check_conda_env(ENV_NAME):
            raise OSError(f"The conda environment {ENV_NAME} not fould, please install {ENV_NAME} first!")

        parent_dir = os.path.dirname(sequence_path)
        time_str = get_cur_time()
        input_folder = f"boltz_input_{time_str}"
        input_dir = os.path.join(parent_dir, input_folder)
        output_dir = os.path.join(parent_dir, f"boltz_output_{time_str}")

        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        with open(sequence_path) as f:
            prots = f.readlines()
        with open(smiles_path) as f:
            smiles = f.readlines()
        name_list = []
        # for i in range(len(prots))[::2]:
        name = prots[0][1:].split()[0]
        name_list.append(name)
        out = f">A|protein\n"
        out += (prots[1].strip() + "\n")
        out += f">B|smiles\n"
        out += (smiles[0].strip() + "\n")
        with open(os.path.join(input_dir, name + ".fasta"), "w") as fw:
            fw.write(out)

        command_parts = []
        command_parts.extend(["conda", "run", "-n", ENV_NAME])

        command_parts.extend([
            "boltz",
            "predict", input_dir,
            "--out_dir", output_dir,
            "--use_msa_server",
            "--msa_server_username", "myuser",
            "--msa_server_password", "mypassword"
        ])

        command = " ".join(command_parts)
        # os.system(command)
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )

        preds_dir = os.path.join(output_dir, f"boltz_results_{input_folder}", "predictions")
        results_list = []
        for name in name_list:
            structure_path = os.path.join(preds_dir, name, f"{name}_model_0.cif")

            if not os.path.exists(structure_path):
                raise ValueError("Results file not exsited.")
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
    ENV_NAME = "clean"
    try:
        if not os.path.exists(sequence_path):
            raise FileNotFoundError(f"Sequence file not found at: {sequence_path}")

        if not os.path.exists(CLEAN_DIR):
            raise FileNotFoundError(f"The CLEAN code directory not found at: {CLEAN_DIR}")

        if not check_conda_env(ENV_NAME):
            raise OSError(f"The conda environment {ENV_NAME} not fould, please install {ENV_NAME} first!")

        parent_dir = os.path.dirname(sequence_path)
        output_dir = os.path.join(parent_dir, "clean_output")

        cur_workspace = os.getcwd()
        os.chdir(CLEAN_DIR)
        command_parts = []
        command_parts.extend(["conda", "run", "-n", ENV_NAME])

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
    ENV_NAME = "catapro"
    try:
        if not os.path.exists(sequence_path):
            raise FileNotFoundError(f"Sequence file not found at: {sequence_path}")

        if not os.path.exists(smiles_path):
            raise FileNotFoundError(f"Molecule smiles file not found at: {smiles_path}")

        if not os.path.exists(CATAPRO_DIR):
            raise FileNotFoundError(f"The CataPro code directory not found at: {CATAPRO_DIR}")

        if not check_conda_env(ENV_NAME):
            raise OSError(f"The conda environment {ENV_NAME} not fould, please install {ENV_NAME} first!")

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
        command_parts.extend(["conda", "run", "-n", ENV_NAME])
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
    ENV_NAME = "prime"
    try:
        if not os.path.exists(sequence_path):
            raise FileNotFoundError(f"Sequence file not found at: {sequence_path}")

        if not os.path.exists(PRIME_DIR):
            raise FileNotFoundError(f"The Prime code directory not found at: {PRIME_DIR}")

        if not check_conda_env(ENV_NAME):
            raise OSError(f"The conda environment {ENV_NAME} not fould, please install {ENV_NAME} first!")

        parent_dir = os.path.dirname(sequence_path)
        output_dir = os.path.join(parent_dir, "prime_output")
        os.makedirs(output_dir, exist_ok=True)

        cur_workspace = os.getcwd()
        os.chdir(PRIME_DIR)
        command_parts = []
        command_parts.extend(["conda", "run", "-n", ENV_NAME])
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
    ENV_NAME = "chroma"
    try:
        if not os.path.exists(structure_path):
            raise FileNotFoundError(f"Structure file not found at: {structure_path}")

        if not os.path.exists(CHROMA_DIR):
            raise FileNotFoundError(f"The Chroma code directory not found at: {CHROMA_DIR}")

        if not check_conda_env(ENV_NAME):
            raise OSError(f"The conda environment {ENV_NAME} not fould, please install {ENV_NAME} first!")

        parent_dir = os.path.dirname(structure_path)
        base_name = os.path.basename(structure_path)
        output_dir = os.path.join(parent_dir, "chroma_output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, base_name)

        cur_workspace = os.getcwd()
        os.chdir(CHROMA_DIR)
        command_parts = []
        command_parts.extend(["conda", "run", "-n", ENV_NAME])
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
    ENV_NAME = "iqtree"
    try:
        if not os.path.exists(msa_path):
            raise FileNotFoundError(f"MSA file not found at: {msa_path}")

        if not check_conda_env(ENV_NAME):
            raise OSError(f"The conda environment {ENV_NAME} not fould, please install {ENV_NAME} first!")

        command_parts = []
        command_parts.extend(["conda", "run", "-n", ENV_NAME])
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
    ENV_NAME = "ligandmpnn_env"
    try:
        if not os.path.exists(structure_path):
            raise FileNotFoundError(f"Structure file not found at: {structure_path}")

        if not os.path.exists(LIGANDMPNN_DIR):
            raise FileNotFoundError(f"The LigandMPNN code directory not found at: {LIGANDMPNN_DIR}")

        if not check_conda_env(ENV_NAME):
            raise OSError(f"The conda environment {ENV_NAME} not fould, please install {ENV_NAME} first!")

        parent_dir = os.path.dirname(structure_path)
        base_name = os.path.basename(structure_path)
        output_dir = os.path.join(parent_dir, "ligandmpnn_output")
        os.makedirs(output_dir, exist_ok=True)

        cur_workspace = os.getcwd()
        os.chdir(LIGANDMPNN_DIR)
        command_parts = []
        command_parts.extend(["conda", "run", "-n", ENV_NAME])
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


@tool
def run_esm_mutation_prediction(sequence_path: str, top_k: int = 20, res_list: Optional[List[int]] = None) -> Dict[
    str, Any]:
    """
    Uses ESM to predict the mutant effect given a protein sequence in FASTA format.
    It can rapidly score all possible mutations in a protein sequence.
    Use Case: Ideal for initial, large-scale virtual screening of mutation sites on
    a protein sequence to quickly identify a smaller set of potentially beneficial
    or detrimental mutations. This is especially useful when a high-resolution structure
     is not available.

     Args:
        sequence_path: Path to the protein sequence in FASTA format.
        top_k: Return the top k mutations (default 20).
        res_list: List of amino acid numbers to be mutated.
            If not provided, then all amino acids are mutated (default None).
    Returns:
        Dictionary containing the top k mutations.
    """
    ESM_DIR = "/home/ubuntu/file2/esm/"
    ENV_NAME = "esm"
    try:
        # TODO: Multi sequence
        if not os.path.exists(sequence_path):
            raise FileNotFoundError(f"Sequence file not found at: {sequence_path}")

        if not os.path.exists(ESM_DIR):
            raise FileNotFoundError(f"The ESM code directory not found at: {ESM_DIR}")

        if not check_conda_env(ENV_NAME):
            raise OSError(f"The conda environment {ENV_NAME} not fould, please install {ENV_NAME} first!")

        seq_list = read_seq_from_fasta(sequence_path)
        seq = seq_list[0][1]

        parent_dir = os.path.dirname(sequence_path)
        output_dir = os.path.join(parent_dir, "esm_output")
        os.makedirs(output_dir, exist_ok=True)

        cur_workspace = os.getcwd()
        os.chdir(ESM_DIR)
        command_parts = []
        command_parts.extend(["conda", "run", "-n", ENV_NAME])
        command_parts.extend([
            "python", "predict_mutations.py",
            "--sequence", seq,
            "--top_k", str(top_k),
            "--save_dir", output_dir,
            "--csv_name", "muts.csv"
        ])
        if res_list:
            command_parts.extend(["--res_list", ",".join([str(res) for res in res_list])])

        command = " ".join(command_parts)

        # os.system(command)
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )

        os.chdir(cur_workspace)

        csv_path = os.path.join(output_dir, "muts.csv")
        with open(csv_path) as f:
            lines = f.readlines()
        # print(lines)
        muts_list = []
        for line in lines[1:]:
            muts, score = line.strip().split(",")
            muts_list.append({
                "mutation": muts,
                "score": score
            })

        return {
            "status": "success",
            "output_path": csv_path,
            "mutations": muts_list
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e)
        }


@tool
def run_prime_mutation_prediction(sequence_path: str, top_k: int = 20, res_list: Optional[List[int]] = None) -> Dict[
    str, Any]:
    """
    Uses Prime to predict the mutant effect given a protein sequence in FASTA format.
    It is designed for rapid evaluation of potential mutations.
    Use Case: Best used for quickly identifying single-site mutations that are likely
    to improve a protein's stability and activity, particularly for applications
    involving high temperatures.

     Args:
        sequence_path: Path to the protein sequence in FASTA format.
        top_k: Return the top k mutations (default 20).
        res_list: List of amino acid numbers to be mutated.
            If not provided, then all amino acids are mutated (default None).
    Returns:
        Dictionary containing the top k mutations.
    """
    PRIME_DIR = "/home/ubuntu/file2/Pro-Prime/"
    ENV_NAME = "prime"
    try:
        if not os.path.exists(sequence_path):
            raise FileNotFoundError(f"Sequence file not found at: {sequence_path}")

        if not os.path.exists(PRIME_DIR):
            raise FileNotFoundError(f"The Pro-Prime code directory not found at: {PRIME_DIR}")

        if not check_conda_env(ENV_NAME):
            raise OSError(f"The conda environment {ENV_NAME} not fould, please install {ENV_NAME} first!")

        parent_dir = os.path.dirname(sequence_path)
        output_dir = os.path.join(parent_dir, "prime_output")
        os.makedirs(output_dir, exist_ok=True)

        # TODO: Multi sequence
        seq_list = read_seq_from_fasta(sequence_path)
        seq = seq_list[0][1]

        cur_workspace = os.getcwd()
        os.chdir(PRIME_DIR)
        command_parts = []
        command_parts.extend(["conda", "run", "-n", ENV_NAME])
        command_parts.extend([
            "python", "predict_mutation.py",
            "--sequence", seq,
            "--top_k", str(top_k),
            "--save_dir", output_dir,
            "--csv_name", "muts.csv"
        ])
        if res_list:
            command_parts.extend(["--res_list", ",".join([str(res) for res in res_list])])
        command = " ".join(command_parts)
        # os.system(command)
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        os.chdir(cur_workspace)

        csv_path = os.path.join(output_dir, "muts.csv")
        with open(csv_path) as f:
            lines = f.readlines()
        # print(lines)
        muts_list = []
        for line in lines[1:]:
            muts, score = line.strip().split(",")
            muts_list.append({
                "mutation": muts,
                "score": score
            })

        return {
            "status": "success",
            "output_path": csv_path,
            "mutations": muts_list
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e)
        }


def extract_amino_acid_info(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    amino_acid_info = []
    seen_residues = set()

    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            for residue in chain:
                if residue.get_resname() not in aa_dict.keys():
                    continue
                if residue.get_id()[0] == ' ' and residue.get_resname() != "HOH":
                    residue_id = (chain_id, residue.get_id()[1])
                    if residue_id not in seen_residues:
                        amino_acid_type = residue.get_resname()
                        amino_acid_number = residue.get_id()[1]
                        amino_acid_info.append((aa_dict[amino_acid_type], amino_acid_number, chain_id))
                        seen_residues.add(residue_id)

    return amino_acid_info


@tool
def run_foldx_mutation_prediction(structure_path: str, top_k: int = 20, res_list: Optional[List[int]] = None) -> Dict[
    str, Any]:
    """
    Uses FoldX to predict the mutant effect given a known protein structure.
    Use Case: A good intermediate tool for analyzing a refined list of mutations
    from a faster initial scan. It provides a good balance of speed and
    accuracy for ranking candidates before moving to more computationally
    intensive methods. It is useful for predicting effects on stability, affinity,
    and specificity.

     Args:
        structure_path: Path to the protein structure in PDB format.
        top_k: Return the top k mutations (default 20).
        res_list: List of amino acid numbers to be mutated.
            If not provided, then all amino acids are mutated (default None).
    Returns:
        Dictionary containing the top k mutations.
    """
    FOLDX_DIR = "/home/ubuntu/file2/FoldX/"
    try:
        if not os.path.exists(structure_path):
            raise FileNotFoundError(f"Structure file not found at: {structure_path}")

        if not os.path.exists(FOLDX_DIR) or not os.path.exists(os.path.join(FOLDX_DIR, "foldx")):
            raise FileNotFoundError(f"The FoldX not found at directory: {FOLDX_DIR}")

        parent_dir = os.path.dirname(structure_path)
        base_name = os.path.basename(structure_path)
        shutil.copy(structure_path, FOLDX_DIR)
        output_dir = os.path.join(parent_dir, "foldx_output")
        os.makedirs(output_dir, exist_ok=True)

        info = extract_amino_acid_info(structure_path)

        muts = ""

        for aa, idx, chain in info:
            if not res_list or idx in res_list:
                muts += f"{aa}{chain}{idx}a,"
        if not muts:
            raise ValueError("No valid residue.")
        muts = muts[:-1]
        cur_workspace = os.getcwd()
        os.chdir(FOLDX_DIR)
        command_parts = [
            "./foldx",
            "--command=PositionScan",
            f"--pdb=./{base_name}",
            f"--positions={muts}",  # HA386a
            f"--output-dir={output_dir}"
        ]

        command = " ".join(command_parts)
        # os.system(command)
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        os.chdir(cur_workspace)

        output_path = os.path.join(output_dir, f"PS_{base_name.split('.')[0]}_scanning_output.txt")
        with open(output_path) as f:
            lines = f.readlines()

        muts_list = []
        for line in lines:
            muts, score = line.strip().split()
            muts_list.append((muts, float(score)))
        sorted_muts = sorted(muts_list, key=lambda x: -x[1])
        top_k = min(len(sorted_muts), top_k)
        res_list = []

        for mut, score in sorted_muts[:top_k]:
            res_list.append({
                "mutation": mut,
                "score": score
            })

        return {
            "status": "success",
            "output_path": output_path,
            "mutations": res_list
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e)
        }


@tool
def run_rosetta_mutation_prediction(structure_path: str, top_k: int = 20, res_list: Optional[List[int]] = None) -> Dict[
    str, Any]:
    """
    Uses Rosetta to predict the mutant effect given a known protein structure.
    Use Case: Best reserved for the final, most detailed analysis of a small
    number of top candidate mutations identified by faster methods. Its high
    accuracy is valuable for making final decisions on which variants to pursue
    experimentally.

    Args:
        structure_path: Path to the protein structure in PDB format.
        top_k: Return the top k mutations (default 20).
        res_list: List of amino acid numbers to be mutated.
            If not provided, then all amino acids are mutated (default None).
    Returns:
        Dictionary containing the top k mutations.
    """
    ROSETTA_DIR = "/home/ubuntu/file2/Pro-Prime/"
    PYROSETTA_ENV = "rosetta"
    ROSETTA_ENV = "rosetta_ddg"
    ROSETTA_DB = "/home/ubuntu/miniconda3/envs/rosetta_ddg/database/"
    try:
        if not os.path.exists(structure_path):
            raise FileNotFoundError(f"Structure file not found at: {structure_path}")

        if not os.path.exists(ROSETTA_DIR):
            raise FileNotFoundError(f"The Rosetta scripts not found at directory: {ROSETTA_DIR}")

        if not os.path.exists(ROSETTA_DB):
            raise FileNotFoundError(f"The Rosetta database not found at directory: {ROSETTA_DB}")

        if not check_conda_env(PYROSETTA_ENV):
            raise OSError(f"The conda environment {PYROSETTA_ENV} not fould, please install pyrosetta first!")

        if not check_conda_env(ROSETTA_ENV):
            raise OSError(f"The conda environment {ROSETTA_ENV} not fould, please install rosetta first!")

        cur_workspace = os.getcwd()
        os.chdir(ROSETTA_DIR)
        command_parts = []
        command_parts.extend(["conda", "run", "-n", PYROSETTA_ENV])
        command_parts.extend([
            "python", "run_rosetta_relax.py",
            "--pdb_path", structure_path
        ])
        command = " ".join(command_parts)
        if res_list:
            command_parts.extend(["--res_list", ",".join([str(res) for res in res_list])])
        # os.system(command)
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )

        parent_dir = os.path.dirname(structure_path)
        pdb_name, surfix = structure_path.rsplit(".", 1)
        relax_pdb_path = f"{pdb_name}.clean.relaxed.{surfix}"
        mut_file = os.path.join(parent_dir, pdb_name + ".mut")

        ddg_cmd = f"conda run -n {ROSETTA_ENV}\
                cartesian_ddg\
                -database {ROSETTA_DB}\
                -s {relax_pdb_path}\
                -ddg::iterations 5\
                -ddg::score_cutoff 1.0\
                -ddg::dump_pdbs false\
                -ddg::bbnbrs 1\
                -score:weights ref2015_cart\
                -ddg::mut_file {mut_file}\
                -ddg:frag_nbrs 2\
                -ignore_zero_occupancy false\
                -missing_density_to_jump \
                -ddg:flex_bb false\
                -ddg::force_iterations false\
                -fa_max_dis 9.0\
                -ddg::json true\
                -ddg:legacy false"
        os.system(ddg_cmd)
        # subprocess.run(
        #     ddg_cmd,
        #     shell=True,
        #     check=True,
        #     capture_output=True,
        #     text=True
        # )

        pdb_name = os.path.basename(structure_path).split(".")[0]
        with open(f'./{pdb_name}.json', 'r') as file:
            data = json.load(file)
        os.chdir(cur_workspace)

        dg_df = pd.DataFrame()
        dg_df['wildtype'] = [str(data[i]['mutations'][0]['wt']) +
                             str(data[i]['mutations'][0]['pos'])
                             for i in range(len(data))]
        dg_df['mutation'] = [data[i]['mutations'][0]['mut'] for i in range(len(data))]
        dg_df['total_score'] = [data[i]['scores']['total'] for i in range(len(data))]
        dg_df['mutant'] = dg_df['wildtype'] + dg_df['mutation']
        ddg_df = dg_df.groupby('mutant', sort=False, as_index=False).min()
        ddg_df['ddG'] = ddg_df['total_score'] - ddg_df['total_score'][0]
        res = ddg_df.sort_values(by='ddG')

        res_list = []
        top_k = min(min(len(res['mutant']), 50), top_k)
        for mut, ddG in zip(res['mutant'][:top_k], res['ddG'][:top_k]):
            res_list.append({
                "mutation": mut,
                "ddG": ddG
            })

        return {
            "status": "success",
            "output_path": f'./{pdb_name}.json',
            "mutations": res_list
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e)
        }

# if __name__ == "__main__":
#     print(run_esm_mutation_prediction("/home/ubuntu/agents/EpHod/example/test_seq.fasta", res_list=[1, 2, 3]))
# print(read_seq_from_fasta("/home/ubuntu/agents/agent_outputs/enzyme_tool_debug/P04637.fasta"))
# print(extract_seq_from_structure(
#     structure_path ="/home/ubuntu/agents/agent_outputs/enzyme_tool_debug/aaa.pdb"))
#     with open("/home/ubuntu/agents/agent_outputs/enzyme_tool_debug/muts/PS_32_scanning_output.txt") as f:
#         lines = f.readlines()
#     for line in lines:
#         print(line.split())
#     info = extract_amino_acid_info("/home/ubuntu/file2/FoldX/32.pdb")
#     print(info)
#     from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
#     import torch
#
#     aa_list = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
#                "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
#     sequence = "AVRLFIEWLKNGGPSSGAPPPSGEGTFTSDLSKQMEEE"
#     model, alphabet = pretrained.load_model_and_alphabet("esm1v_t33_650M_UR90S_1")
#     model.eval()
#     model = model.cpu()
#     # if torch.cuda.is_available():
#     #     model = model.cuda()
#     batch_converter = alphabet.get_batch_converter()
#     data = [
#         ("protein1", sequence),
#     ]
#     batch_labels, batch_strs, batch_tokens = batch_converter(data)
#     with torch.no_grad():
#         token_probs = torch.log_softmax(model(batch_tokens.cpu())["logits"], dim=-1)
#     print(len(sequence), token_probs.shape)
#     mut_list = []
#     for i, aa in enumerate(sequence):
#         ori_idx = alphabet.get_idx(aa)
#         for mt in aa_list:
#             if aa == mt:
#                 continue
#             mt_idx = alphabet.get_idx(mt)
#             score = token_probs[0, 1 + i, mt_idx] - token_probs[0, 1 + i, ori_idx]
#             mut_list.append((f"{aa}{i}{mt}", score))
#     sorted_muts = sorted(mut_list, key=lambda x: -x[1])
#     print(sorted_muts[:20])
# print(token_probs.shape)

#     input_fasta = "/home/ubuntu/agents/agent_outputs/enzyme_tool_debug/query.fasta"
#     os.chdir("/home/ubuntu/file2/CLEAN/app/")
#     os.system(
#         "conda run -n clean python CLEAN_infer_fasta.py --fasta_data '/home/ubuntu/agents/agent_outputs/enzyme_tool_debug/query.fasta'")

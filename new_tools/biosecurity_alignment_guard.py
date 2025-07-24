
# new_tools/biosecurity_alignment_guard.py

from smolagents import tool
import subprocess
from new_tools._core_guard.keyword_filter import keyword_check
from new_tools._core_guard.embedding_filter import embedding_check
from new_tools._core_guard.sequence_scanner import sequence_check

@tool
def biosecurity_alignment_guard(text: str, mode: str = "strict") -> str:
    """
    Biosecurity alignment guard with multi-layer detection.

    Args:
        text (str): Text content to be scanned (e.g., model input/output).
        mode (str): Risk response mode. "strict" will redact risky content; "warn" will annotate it.

    Returns:
        str: Either the original text (if clean), a redacted message, or a warning-tagged version.
    """
    risky = any([
        keyword_check(text),
        embedding_check(text),
        sequence_check(text)
    ])

    if not risky:
        return text

    if mode == "strict":
        return "[REDACTED: BIOSECURITY VIOLATION]"
    else:
        return f"[⚠️ BIO RISK] {text}"
    

@tool
def run_jailbreak_hiv(config_path: str, output_log: str = "/home/ubuntu/agent_outputs/jailbreak_output.log") -> str:
    """
    Run the GeneBreaker jailbreak HIV script using the evo2 Conda environment.

    Args:
        config_path: Path to the JSON config file
        output_log: File path to save the output log

    Returns:
        Execution result or error message
    """
    try:
        command = f"""
	export CUDA_VISIBLE_DEVICES=1 && \
	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
	source ~/miniconda3/etc/profile.d/conda.sh && \
	conda activate evo2 && \
	python /home/ubuntu/GeneBreaker/auto_jailbreak_hiv.py \
 	  --model_name evo2_7b \
	  --skip_chatgpt \
 	  --accession U63632.1 \
	  --gene_focus env
	"""





        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=300)

        with open(output_log, "w") as f:
            f.write(result.stdout + "\n\n" + result.stderr)

        if result.returncode != 0:
            return f"❌ Error occurred (exit code {result.returncode}). Check log at {output_log}"

        return f"✅ Success. Log saved to {output_log}"

    except Exception as e:
        return f"❌ Exception: {str(e)}"



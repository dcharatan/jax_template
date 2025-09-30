import getpass
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

USERNAME = os.getlogin() or getpass.getuser()
JOB_ROOT = Path(f"/data/scene-rep/u/{USERNAME}/jobs/jax_template")
SCRIPT = "python3 -m main"
ENVIRONMENT_NAME = "jax_template"


# Override these with environment variables.
DEFAULT_BRANCH = "main"
DEFAULT_COMMIT_SHA = None
DEFAULT_CHECKPOINT = None
DEFAULT_QOS = "shared-if-available"
DEFAULT_PARTITIONS = ",".join(
    (
        "vision-shared-rtx3080",  # 10 GB
        "vision-shared-rtx2080ti",  # 11 GB
        "vision-shared-titanv",  # 12 GB
        "vision-shared-titanrtx",  # 24 GB
        "vision-shared-rtx3090",  # 24 GB
        "vision-shared-rtx4090",  # 24 GB
        "vision-shared-v100",  # 32 GB
        "vision-shared-l40s",  # 48 GB
        "vision-shared-a6000",  # 48 GB
        "vision-shared-rtx6000ada",  # 48 GB
        "vision-shared-a100",  # 80 GB
        "vision-shared-h100",  # 80 GB or 96 GB
        "vision-shared-h200",  # 141 GB
    )
)
DEFAULT_NUM_GPUS = 1
DEFAULT_MEMORY = 32
DEFAULT_TIME = "24:00:00"

# Define color codes.
CYAN = "\x1b[36m"
RESET = "\x1b[39m"


def run_command_on_login_node(command: str) -> str:
    if platform.node() != "slurm-login-1":
        # Properly quote the remote command
        remote_command = shlex.quote(command)
        command = f"ssh {USERNAME}@slurm-login-1 {remote_command}"
        return subprocess.run(
            command, shell=True, capture_output=True, text=True, check=True
        ).stdout
    else:
        return subprocess.run(
            command, shell=True, capture_output=True, text=True, check=True
        ).stdout


def get_timestamp() -> str:
    return datetime.now().strftime("%Y_%m_%d.%H_%M_%S")


def message(message: str) -> None:
    print(f"{CYAN}{message}{RESET}")


def job_name_from_script(script: str) -> str:
    # Replace any character that is not alphanumeric, hyphen, underscore, or period
    # with an underscore.
    name = re.sub(r"[^a-zA-Z0-9\-_\.]", "_", script)

    # Add a timestamp.
    return f"{get_timestamp()}.{name}"


def submit_job(
    script: str,
    branch: str = DEFAULT_BRANCH,
    commit_sha: str | None = DEFAULT_COMMIT_SHA,
    checkpoint: str | None = DEFAULT_CHECKPOINT,
    partitions: str = DEFAULT_PARTITIONS,
    qos: str = DEFAULT_QOS,
    num_gpus: int = DEFAULT_NUM_GPUS,
    memory: int = DEFAULT_MEMORY,
    time: str = DEFAULT_TIME,
    sweep_path: str | None = None,
) -> Path:
    message(f"SCRIPT: {script}")
    message(f"PARTITIONS: {partitions}")
    message(f"QOS: {qos}")
    message(f"NUM_GPUS: {num_gpus}")
    message(f"MEMORY: {memory}GB")
    message(f"TIME: {time}")
    message("---")

    # Extract a run name.
    job_name = job_name_from_script(script)

    # Pick a job directory.
    if sweep_path is not None:
        job_dir = JOB_ROOT / sweep_path / job_name
    else:
        job_dir = JOB_ROOT / job_name
    job_dir.mkdir(parents=True, exist_ok=True)

    # Clone the code.
    code_dir = job_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)

    # Create a symlink to the latest run.
    latest_run = JOB_ROOT / "latest"
    latest_run.unlink(missing_ok=True)
    latest_run.symlink_to(job_dir, target_is_directory=True)

    # Pull the repos.
    message(f"Cloning code from branch {branch}.")
    code_dir = code_dir / "code"
    os.system(
        "git clone --depth 1 --single-branch --no-tags "
        f"--branch {branch} git@github.com:dcharatan/jax_template.git "
        f"{code_dir}"
    )

    # Check out a specific commit.
    if commit_sha is not None:
        print(f"Checking out {commit_sha} in {code_dir}.")
        os.system(
            f"cd {code_dir} && "
            f"git fetch origin {commit_sha} && "
            f"git checkout {commit_sha} && "
            f"cd -"
        )

    # Create the Slurm job file.
    workspace_dir = job_dir / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # If checkpoint is not None, copy it to the new dir workspace_dir / checkpoints.
    if checkpoint is not None:
        checkpoint_dir = workspace_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        message("Copying checkpoint.")
        shutil.copytree(checkpoint, checkpoint_dir / Path(checkpoint).name)

    cuda_cache_dir = job_dir / "cuda_cache"
    cuda_cache_dir.mkdir(parents=True, exist_ok=True)
    torch_extensions_cache_dir = job_dir / ".cache/torch_extensions"
    torch_extensions_cache_dir.mkdir(parents=True, exist_ok=True)
    slurm_file = f"""#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -o {job_dir}/out.txt
#SBATCH -e {job_dir}/error.txt
#SBATCH --open-mode=append
#SBATCH --mail-user={USERNAME}
#SBATCH --mail-type=FAIL
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={4}
#SBATCH --mem={memory * num_gpus}G
#SBATCH --partition={partitions}
#SBATCH --qos={qos}
#SBATCH --account vision-sitzmann
#SBATCH --time={time}
#SBATCH --signal=SIGTERM@30
#SBATCH --requeue

source /data/scene-rep/u/{USERNAME}/miniconda3/etc/profile.d/conda.sh
cd {code_dir.resolve()}
conda activate {ENVIRONMENT_NAME}

export CUDA_CACHE_PATH={cuda_cache_dir.resolve()}
export TORCH_EXTENSIONS_DIR={torch_extensions_cache_dir.resolve()}
export WORKSPACE={workspace_dir.resolve()}

{script}
"""  # noqa: E501

    slurm_script_path = job_dir / "job.slurm"
    with slurm_script_path.open("w") as f:
        f.write(slurm_file)

    os.system(f"chmod +x {slurm_script_path}")
    run_command_on_login_node(f"sbatch {slurm_script_path}")
    message(f"SCRIPT: {script}")
    message(f"WORKSPACE: {workspace_dir}")
    return job_dir


if __name__ == "__main__":
    if __package__ is not None:
        print("This script must be run directly (not as a module).")
        sys.exit(1)

    # Send the most relevant information about the job to the user.
    script = f"{SCRIPT} {' '.join(sys.argv[1:])}".strip()
    submit_job(
        script,
        branch=os.environ.get("BRANCH", DEFAULT_BRANCH),
        commit_sha=os.environ.get("COMMIT_SHA", DEFAULT_COMMIT_SHA),
        checkpoint=os.environ.get("CHECKPOINT", DEFAULT_CHECKPOINT),
        partitions=os.environ.get("PARITITONS", DEFAULT_PARTITIONS),
        qos=os.environ.get("QOS", DEFAULT_QOS),
        num_gpus=int(os.environ.get("NUM_GPUS", DEFAULT_NUM_GPUS)),
        memory=int(os.environ.get("MEMORY", DEFAULT_MEMORY)),
        time=os.environ.get("TIME", DEFAULT_TIME),
    )

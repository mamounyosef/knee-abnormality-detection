"""upload_model.py - upload the exported folder to the Hugging Face repo.

One-time login first (paste a WRITE token from https://huggingface.co/settings/tokens):
    hf auth login

Then:
    python hf_release/export_model.py      # builds hf_release/out/<run>/
    python hf_release/upload_model.py      # uploads it

What it does:
    hf_release/out/<run>/  ->  <repo>/<run>/   (one folder per model inside one repo)
The upload is one commit in the repo's history, like a git commit.
"""
from pathlib import Path

from huggingface_hub import HfApi   # the official Hugging Face client

REPO_ID = "mamounyosef/knee-abnormality-detection"    # <username>/<repo name>
RUN = "convnext_base_v5_96slices_per_finding_attn"    # the model folder to upload
HERE = Path(__file__).resolve().parent                # hf_release/


def main():
    api = HfApi()                                      # uses the token saved by `hf auth login`
    print("logged in as", api.whoami()["name"])        # fails early if not logged in

    folder = HERE / "out" / RUN
    if not (folder / "model.safetensors").exists():
        raise SystemExit(f"{folder} is not exported yet: run hf_release/export_model.py first")

    # the whole model folder, placed under <repo>/<run>/ (the repo front page is added later, with all models)
    # big files like model.safetensors go to Hugging Face's large-file storage automatically
    api.upload_folder(folder_path=str(folder), path_in_repo=RUN, repo_id=REPO_ID, repo_type="model",
                      commit_message=f"Add {RUN}", ignore_patterns=["__pycache__/*", "*.pyc"])
    print(f"done: https://huggingface.co/{REPO_ID}/tree/main/{RUN}")


if __name__ == "__main__":
    main()

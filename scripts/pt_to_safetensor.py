import argparse
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO = "stevenhe04/deep-mca"


def convert_local(pt_path: Path) -> None:
    out_path = pt_path.with_suffix(".safetensors")
    state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
    save_file(state_dict, out_path)
    print(f"converted {pt_path} to {out_path}")


def convert_hub(repo_id: str, arch: str, revision: str) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    weights_path = hf_hub_download(repo_id=repo_id, filename=f"{arch}/model.pt", revision=revision)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    tmp_dir = tempfile.mkdtemp()

    try:
        out_path = Path(tmp_dir) / "model.safetensors"
        save_file(state_dict, out_path)

        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo=f"{arch}/model.safetensors",
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
        )

        print(f"uploaded {arch}/model.safetensors to {repo_id}")
    finally:
        shutil.rmtree(tmp_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=Path)
    parser.add_argument("--arch", type=str, default="skylake")
    parser.add_argument("--revision", type=str, default="main")
    args = parser.parse_args()

    if args.local:
        convert_local(args.local)
    else:
        convert_hub(REPO, args.arch, args.revision)


if __name__ == "__main__":
    main()

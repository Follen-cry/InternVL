#!/usr/bin/env python
"""
Assemble a full InternVL-U unified pipeline directory by combining:
  - your finetuned (or LoRA-merged) VLM checkpoint
  - the unchanged generation_decoder / vae / scheduler / processor / etc.
    from the original HuggingFace snapshot
into a single directory that `InternVLUPipeline.from_pretrained(...)` can load.

Subcomponents from the snapshot are exposed via symlinks (no extra disk use).
The replaced VLM is copied (so you own the bytes and can ship it independently).
Top-level pipeline files (model_index.json, README, etc.) are copied as well.

Usage:
    python tools/assemble_unified.py \
        --vlm     /scratch/network/ssd2/junlin/models/internvlu/internvl-u-4epoch-merged \
        --output  /scratch/network/ssd2/junlin/models/internvlu/internvl-u-4-epoch-full-finetuned

The original snapshot is auto-detected from the HuggingFace cache. Override with
--snapshot /path/to/snapshot if needed (the dir that contains model_index.json).

Add --copy-vlm-as-symlink to symlink the new VLM as well (saves disk; loses
portability — moving the output dir without the source breaks the pipeline).
Add --force to overwrite an existing output directory.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

DEFAULT_REPO_ID = "InternVL-U/InternVL-U"


def find_snapshot(repo_id: str) -> Path:
    """Locate the latest local snapshot of `repo_id` in the HF cache."""
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo_id=repo_id, local_files_only=True)
        return Path(path)
    except Exception as e:
        sys.exit(
            f"Could not auto-detect HF snapshot for '{repo_id}': {e}\n"
            f"Pass --snapshot /path/to/snapshot explicitly."
        )


def list_subcomponents(snapshot: Path):
    """Read model_index.json and return [(name, is_dir)] for each subcomponent."""
    idx = snapshot / "model_index.json"
    if not idx.is_file():
        sys.exit(f"No model_index.json at {snapshot} — not a DiffusionPipeline checkpoint.")
    with idx.open() as f:
        manifest = json.load(f)
    components = []
    for k, v in manifest.items():
        if k.startswith("_"):
            continue
        # entry is typically [library_name, class_name]
        if isinstance(v, list) and len(v) == 2:
            sub = snapshot / k
            components.append((k, sub.is_dir()))
    return components


def copy_top_level(snapshot: Path, out: Path):
    """Copy small top-level files (configs, manifests, READMEs) into the output."""
    copied = []
    for item in snapshot.iterdir():
        if item.is_dir():
            continue
        # Don't copy .lock or partial files
        if item.name.startswith(".") or item.suffix in (".lock", ".incomplete"):
            continue
        dst = out / item.name
        # cp -L through symlink to get the real bytes; tiny files anyway
        shutil.copyfile(item.resolve(), dst)
        copied.append(item.name)
    return copied


def link_or_copy_dir(src: Path, dst: Path, *, copy: bool):
    """Symlink (default) or recursively copy a subcomponent directory."""
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    if copy:
        shutil.copytree(src.resolve(), dst, symlinks=False)
    else:
        os.symlink(src.resolve(), dst)  # absolute target — robust to cwd


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--vlm", required=False, type=Path, default=None,
                    help="Finetuned/merged VLM directory (has its own config.json + model.safetensors). "
                         "Required unless --from-unified is provided.")
    ap.add_argument("--output", required=True, type=Path,
                    help="Where to write the assembled unified pipeline")
    ap.add_argument("--snapshot", type=Path, default=None,
                    help="Original HF snapshot dir (auto-detected if omitted)")
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                    help=f"HF repo id used for snapshot auto-detection (default: {DEFAULT_REPO_ID})")
    ap.add_argument("--replace", default="vlm",
                    help="Name of the subcomponent to replace with --vlm (default: vlm)")
    ap.add_argument("--copy-vlm-as-symlink", action="store_true",
                    help="Symlink the VLM instead of copying its bytes (saves disk, less portable)")
    ap.add_argument("--from-unified", type=Path, default=None,
                    help=(
                        "Path to a directory written by InternVLUUnifiedModel.save_pretrained "
                        "(or merge_lora_u_full.py). When given, EVERY subcomponent that exists "
                        "under --from-unified is taken from there; only subcomponents that "
                        "are missing fall back to the snapshot. --vlm and --replace continue "
                        "to take precedence for the named subcomponent."
                    ))
    ap.add_argument("--force", action="store_true",
                    help="Overwrite --output if it already exists")
    args = ap.parse_args()

    if args.vlm is None and args.from_unified is None:
        sys.exit("Either --vlm or --from-unified must be provided.")

    unified_src: Path = args.from_unified.resolve() if args.from_unified is not None else None
    if unified_src is not None and not unified_src.is_dir():
        sys.exit(f"--from-unified path is not a directory: {unified_src}")

    if args.vlm is not None:
        vlm_src: Path = args.vlm.resolve()
        if not vlm_src.is_dir():
            sys.exit(f"--vlm path is not a directory: {vlm_src}")
        if not (vlm_src / "config.json").is_file():
            sys.exit(f"--vlm dir is missing config.json: {vlm_src}")
    else:
        # Fall back to the unified dir's vlm subdir.
        vlm_candidate = unified_src / args.replace if unified_src is not None else None
        if vlm_candidate is None or not vlm_candidate.is_dir():
            sys.exit(
                f"--vlm not provided and --from-unified does not contain a {args.replace}/ subdir."
            )
        vlm_src = vlm_candidate.resolve()

    snapshot = (args.snapshot or find_snapshot(args.repo_id)).resolve()
    if not (snapshot / "model_index.json").is_file():
        sys.exit(f"--snapshot dir has no model_index.json: {snapshot}")

    out: Path = args.output.resolve()
    if out.exists():
        if not args.force:
            sys.exit(f"--output exists: {out}. Pass --force to overwrite.")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    print(f"[ok] snapshot   : {snapshot}")
    print(f"[ok] new vlm    : {vlm_src}")
    print(f"[ok] output     : {out}")

    # 1. Top-level files (model_index.json, etc.) — copied so we own them.
    copied = copy_top_level(snapshot, out)
    print(f"[ok] copied top-level: {sorted(copied)}")

    # 2. Subcomponents from the manifest.
    components = list_subcomponents(snapshot)
    print(f"[ok] subcomponents: {[name for name, _ in components]}")

    for name, is_dir in components:
        src_sub = snapshot / name
        dst_sub = out / name
        if not is_dir:
            # Some pipelines list non-directory components (rare). Skip with note.
            print(f"[skip] {name}: not a directory under snapshot")
            continue

        if name == args.replace:
            link_or_copy_dir(vlm_src, dst_sub, copy=not args.copy_vlm_as_symlink)
            mode = "symlinked" if args.copy_vlm_as_symlink else "copied"
            print(f"[new ] {name}: {mode} from {vlm_src}")
            continue

        if unified_src is not None and (unified_src / name).is_dir():
            # Take this subcomponent from the unified training output rather than the snapshot.
            link_or_copy_dir((unified_src / name).resolve(), dst_sub, copy=False)
            print(f"[link] {name} -> {(unified_src / name).resolve()}  (from --from-unified)")
        else:
            link_or_copy_dir(src_sub, dst_sub, copy=False)
            print(f"[link] {name} -> {src_sub}")

    # 3. Sanity check — the replaced component must contain real files
    target = out / args.replace
    if not (target / "config.json").is_file():
        sys.exit(f"[error] {target}/config.json is missing — assembly failed")
    weights = list(target.glob("*.safetensors")) + list(target.glob("*.bin"))
    if not weights:
        sys.exit(f"[error] no weight files found in {target}")
    print(f"[ok] {args.replace} contains {len(weights)} weight file(s)")

    print("\nDone. Try:")
    print(f"  python -c \"from internvlu import InternVLUPipeline; "
          f"import torch; "
          f"p = InternVLUPipeline.from_pretrained('{out}', torch_dtype=torch.bfloat16); "
          f"print('loaded ok')\"")


if __name__ == "__main__":
    main()
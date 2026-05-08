"""Merge LoRA adapters in a *full unified* InternVL-U checkpoint directory.

Equivalent of ``tools/merge_lora_u.py`` for the full-unified pipeline layout
written by :meth:`InternVLUUnifiedModel.save_pretrained`. Unlike the original
(which only handles the VLM half), this script:

* merges LoRA on the LLM (and the ViT backbone if it was trained with LoRA);
* if the ``generation_decoder/`` subdir contains LoRA adapters, merges those
  too via :meth:`peft.PeftModel.merge_and_unload`;
* preserves the pipeline directory layout so the result is still loadable by
  ``InternVLUPipeline.from_pretrained``.

The non-replaced subcomponents (vae/, scheduler/, processor/) are symlinked
into the output directory to save disk; pass ``--copy`` to copy bytes instead.

Usage::

    python tools/merge_lora_u_full.py \
        /path/to/internvl-u-trained/ \
        /path/to/internvl-u-trained-merged/
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import torch

# Ensure the local internvl modules are importable regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure the inference-side internvlu package is importable so we can save
# decoder LoRA back through the published class.
_INTERNVLU_PKG = os.environ.get(
    "INTERNVLU_PKG_PATH", "/scratch/network/ssd2/junlin/ssl_mllm/InternVL-U"
)
if _INTERNVLU_PKG and os.path.isdir(_INTERNVLU_PKG) and _INTERNVLU_PKG not in sys.path:
    sys.path.insert(0, _INTERNVLU_PKG)

from internvl.model.internvlu import InternVLUChatModel  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def _link_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    if copy:
        shutil.copytree(src.resolve(), dst, symlinks=False)
    else:
        os.symlink(src.resolve(), dst)


def merge_vlm(in_vlm: Path, out_vlm: Path) -> None:
    """Load the trained VLM, merge any LoRA adapters, save to ``out_vlm``."""
    print(f"[vlm] loading from {in_vlm}")
    model = InternVLUChatModel.from_pretrained(
        str(in_vlm), low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
    ).eval()
    if getattr(model.config, "use_backbone_lora", 0):
        model.vision_model.merge_and_unload()
        model.vision_model = model.vision_model.model
        model.config.use_backbone_lora = 0
    if getattr(model.config, "use_llm_lora", 0):
        model.language_model.merge_and_unload()
        model.language_model = model.language_model.model
        model.config.use_llm_lora = 0
    out_vlm.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_vlm))
    tok = AutoTokenizer.from_pretrained(str(in_vlm), trust_remote_code=False)
    tok.save_pretrained(str(out_vlm))
    print(f"[vlm] wrote merged to {out_vlm}")


def merge_gen_decoder(in_dec: Path, out_dec: Path) -> None:
    """Merge any LoRA adapters on the generation_decoder.

    Falls back to a plain copy if no adapters are present.
    """
    has_adapter = any(p.name.startswith("adapter_") for p in in_dec.iterdir())
    if not has_adapter:
        print(f"[gen_decoder] no LoRA adapter under {in_dec}; copying as-is")
        _link_or_copy(in_dec, out_dec, copy=True)
        return

    print(f"[gen_decoder] merging LoRA adapters under {in_dec}")
    import internvlu.diffusion as ivu_diff
    from peft import PeftModel

    base = ivu_diff.InternVLUGenerationDecoder.from_pretrained(
        str(in_dec), torch_dtype=torch.bfloat16
    )
    peft_model = PeftModel.from_pretrained(base, str(in_dec))
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(str(out_dec))
    print(f"[gen_decoder] wrote merged to {out_dec}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_path", type=Path, help="trained pipeline dir")
    ap.add_argument("output_path", type=Path, help="destination merged pipeline dir")
    ap.add_argument(
        "--copy",
        action="store_true",
        help="Copy bytes for non-replaced subcomponents instead of symlinking.",
    )
    args = ap.parse_args()

    in_dir: Path = args.input_path.resolve()
    out_dir: Path = args.output_path.resolve()

    if not (in_dir / "vlm").is_dir():
        sys.exit(f"--input_path missing vlm/ subdir: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    merge_vlm(in_dir / "vlm", out_dir / "vlm")

    if (in_dir / "generation_decoder").is_dir():
        merge_gen_decoder(in_dir / "generation_decoder", out_dir / "generation_decoder")

    for sub in ("vae", "scheduler", "processor"):
        src = in_dir / sub
        if src.is_dir() and not (out_dir / sub).exists():
            _link_or_copy(src, out_dir / sub, copy=args.copy)
            mode = "copied" if args.copy else "linked"
            print(f"[{sub}] {mode} from {src}")

    idx = in_dir / "model_index.json"
    if idx.is_file():
        shutil.copyfile(idx, out_dir / "model_index.json")
        print(f"[model_index] copied")

    print(f"\nDone. Merged pipeline at: {out_dir}")


if __name__ == "__main__":
    main()

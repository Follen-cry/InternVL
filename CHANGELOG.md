# Changelog — feature/full-unified-sft

## Summary

Adds end-to-end SFT for the InternVL-U **full unified pipeline** (VLM +
generation_decoder + VAE + scheduler + processor) on top of the existing
VLM-only LoRA recipe. The existing scripts and modules are untouched
behaviour-wise; every change is additive or gated on a new flag.

## New files

* ``internvl_chat/internvl/model/internvlu/modeling_internvlu_unified.py`` —
  ``InternVLUUnifiedModel`` (composes vlm + generation_decoder + vae +
  scheduler + processor), with ``forward`` that routes by ``task_type``
  to text-CE or flow-matching MSE, ``save_pretrained`` that writes the
  pipeline directory layout, and ``_build_gen_conditioning`` that
  replicates the inference-time conditioning contract.
* ``internvl_chat/internvl/model/internvlu/flow_matching_utils.py`` —
  SD3-style logit-normal flow-matching schedule helpers
  (``compute_density_for_timestep_sampling``, ``shift_sigma``,
  ``build_noisy_latents``, ``compute_loss_weighting``,
  ``sigma_to_timestep``).
* ``internvl_chat/internvl/model/internvlu/UNIFIED_DESIGN_NOTES.md`` —
  Phase-0 discovery notes (snapshot anatomy, conditioning contract
  quoted from inference code, flow-matching recipe, save/load layout,
  caveats).
* ``internvl_chat/internvl/model/internvlu/UNIFIED_VERIFICATION.md`` —
  Phase-6 verification record.
* ``internvl_chat/internvl/train/dataset_unified.py`` —
  ``ImgenLazyDataset`` (reads ``{task_type, caption, target_image}``
  rows), ``TaskTypeBatchSampler`` (homogeneous-task batches at
  ``imgen_ratio``), ``UnifiedCollator`` (delegates understanding to
  ``concat_pad_data_collator`` so old behaviour is bit-identical;
  builds imgen branch with ``cfg_dropout``).
* ``internvl_chat/internvl/train/internvl_chat_finetune_u_full.py`` —
  new training entrypoint mirroring ``internvl_chat_finetune_u.py``.
* ``internvl_chat/shell/internvlu/internvlu_4b_sft_full_unified.sh`` —
  driver shell script.
* ``internvl_chat/tools/merge_lora_u_full.py`` — merges LoRA on both
  the VLM and (if present) the generation_decoder while preserving the
  pipeline directory layout.
* ``internvl_chat/tests/`` — pytest suite (20 tests covering
  flow-matching math, dataset/sampler/collator, freeze rules, forward
  routing, save/load roundtrip).
* ``internvl_chat/tests/fixtures/`` — synthetic 4 understanding +
  4 imgen meta JSON, JSONLs, and tiny coloured images.

## Modified files (additive only)

* ``internvl_chat/internvl/model/internvlu/__init__.py`` — adds exports
  for ``InternVLUUnifiedConfig``, ``InternVLUUnifiedModel``,
  ``UnifiedForwardOutput``. Existing exports unchanged.
* ``internvl_chat/tools/assemble_unified.py`` — adds ``--from-unified``
  flag. When omitted, the default behaviour is unchanged (snapshot
  links + ``--vlm`` replacement). When given, every subcomponent that
  exists under that directory is taken from there; the remaining ones
  fall back to the snapshot.

## New flags (full-unified shell script / training entrypoint)

| Flag                       | Default | Notes                                                                 |
|---------------------------|---------|-----------------------------------------------------------------------|
| ``freeze_gen_decoder``    | ``False``  | Whether to skip training the generation_decoder.                  |
| ``freeze_vae``            | ``True``   | VAE frozen by default (matches published recipes).                |
| ``gen_decoder_lr``        | ``5e-5``   | LR applied to the ``gen_decoder`` param group.                    |
| ``gen_loss_weight``       | ``0.1``    | Final weight on the flow-matching MSE after warmup.               |
| ``gen_loss_warmup_steps`` | ``1000``   | Linear ramp from 0 → ``gen_loss_weight`` over this many steps.    |
| ``imgen_ratio``           | ``0.3``    | Fraction of imgen-only batches per epoch.                         |
| ``cfg_dropout``           | ``0.1``    | Probability of replacing imgen prompts with ``<img_uncond>``.     |
| ``gen_image_size``        | ``1024``   | Target square resolution before VAE encode.                       |
| ``gen_max_seq_length``    | ``768``    | Tokenizer truncation cap for imgen prompts.                       |
| ``snapshot_dir``          | (auto)     | Pipeline snapshot dir override (auto-detected from ``model_name_or_path``). |

## Meta-JSON schema (additive)

Existing meta entries continue to load unchanged — they are treated as
``task_type="understanding"`` by default. New imgen entries look like:

```json
{
  "synth_imgen": {
    "root": "/path/to/images",
    "annotation": "/path/to/imgen.jsonl",
    "task_type": "imgen",
    "data_augment": false,
    "max_dynamic_patch": 1,
    "repeat_time": 1,
    "length": 4
  }
}
```

with imgen JSONL rows of the form
``{"id": int, "task_type": "imgen", "caption": str, "target_image": path}``.

## How to run

* **Old VLM-only LoRA SFT** — unchanged:

  ```bash
  bash internvl_chat/shell/internvlu/internvlu_4b_sft_lora.sh
  ```

* **New full-unified SFT**:

  ```bash
  INTERNVLU_CKPT=<snapshot>/vlm \
  META_PATH=<your meta.json> \
  OUTPUT_DIR=<out> \
  GPUS=8 PER_DEVICE_BATCH_SIZE=1 BATCH_SIZE=64 \
  bash internvl_chat/shell/internvlu/internvlu_4b_sft_full_unified.sh
  ```

* **Merge LoRA on a unified output**:

  ```bash
  python internvl_chat/tools/merge_lora_u_full.py <out> <out>-merged
  ```

* **Assemble a runnable pipeline directory from a unified output**:

  ```bash
  python internvl_chat/tools/assemble_unified.py \
    --from-unified <out>-merged \
    --output      <out>-pipeline
  ```

* **Tests**:

  ```bash
  cd internvl_chat && python -m pytest tests/ -x
  ```

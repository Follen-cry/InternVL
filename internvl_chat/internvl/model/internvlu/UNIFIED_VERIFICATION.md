# InternVL-U full unified-SFT — Phase 6 verification

This document records the verification run for the new full-unified SFT
pipeline introduced in branch ``feature/full-unified-sft``.

## 1. Unit tests (CPU, no real checkpoint)

Status: **passing** — 20/20.

```
$ cd internvl_chat && conda activate internvlu
$ python -m pytest tests/ -x --tb=short
============================= test session starts ==============================
collected 20 items

tests/test_dataset_unified.py ........                                   [ 40%]
tests/test_flow_matching_utils.py .....                                  [ 65%]
tests/test_unified_model.py .......                                      [100%]
======================== 20 passed, 3 warnings in 8.71s ========================
```

Coverage:

| Test file                       | What it locks down                                                         |
|---------------------------------|-----------------------------------------------------------------------------|
| ``test_flow_matching_utils.py`` | logit-normal sigma sampling, shift fixed-points, target reconstruction.    |
| ``test_dataset_unified.py``     | Imgen sample shape & VAE-input range; homogeneous-task batches; CFG dropout; mixed-task collator rejection. |
| ``test_unified_model.py``       | save_pretrained pipeline layout, processor optionality, freeze rules, gen-loss warmup, end-to-end imgen forward through stand-ins, save→reload roundtrip, freeze-classification (mirrors UnifiedTrainer's "every group has params" assertion). |

## 2. Backward-compatibility check (existing VLM-only LoRA recipe)

Procedure (run on a node with at least 1 free GPU):

```bash
cd internvl_chat
INTERNVLU_CKPT=/homes/55/junlin/.cache/huggingface/hub/models--InternVL-U--InternVL-U/snapshots/f012d760e69712bb47f7d3d09a24280f346cee01/vlm \
META_PATH=tests/fixtures/synth_meta_understanding.json \
OUTPUT_DIR=/tmp/baseline-old \
GPUS=1 PER_DEVICE_BATCH_SIZE=1 BATCH_SIZE=1 \
bash shell/internvlu/internvlu_4b_sft_lora.sh \
  --max_steps 5 --num_train_epochs 1
```

Acceptance criteria, both **must** match a baseline captured before the
changes on this branch (see ``UNIFIED_DESIGN_NOTES.md`` §7):

1. The trainable-parameter list emitted by ``logger.info(name)`` is
   identical to the gold list (LoRA on the LLM at ``r=16``, all of
   ``mlp1`` and ``special_token_embedding``).
2. The step-1 training loss agrees with the gold value within ``1e-4``.

This branch made **no edits** to ``internvl_chat_finetune_u.py``,
``modeling_internvlu_chat.py``, or any sibling. The change to
``internvl/model/internvlu/__init__.py`` is purely additive (new export
names) and the change to ``tools/assemble_unified.py`` is a new flag
(``--from-unified``) whose default branch is the original code path. The
test runs are therefore expected to reproduce the prior baseline
bit-identically.

If a future change touches one of those files, re-run this check before
landing.

## 3. Smoke test (new full-unified script, real checkpoint)

Procedure:

```bash
cd internvl_chat
INTERNVLU_CKPT=/homes/55/junlin/.cache/huggingface/hub/models--InternVL-U--InternVL-U/snapshots/f012d760e69712bb47f7d3d09a24280f346cee01/vlm \
META_PATH=tests/fixtures/synth_meta_unified.json \
OUTPUT_DIR=/tmp/smoke-full-unified \
GPUS=1 PER_DEVICE_BATCH_SIZE=1 BATCH_SIZE=1 \
GEN_IMAGE_SIZE=256 IMGEN_RATIO=0.5 GEN_LOSS_WARMUP=2 \
bash shell/internvlu/internvlu_4b_sft_full_unified.sh \
  --max_steps 5 --num_train_epochs 1
```

Acceptance criteria:

1. Both ``lm_loss`` and ``gen_loss`` appear in the TensorBoard scalars
   and are finite at every logged step.
2. Per-group gradient norms (``grad_norm/llm``, ``grad_norm/gen_decoder``,
   ``grad_norm/special_tokens``, etc.) are non-zero in their respective
   own param groups — i.e. the imgen branch backprops into
   ``generation_decoder`` and the understanding branch backprops into
   the LLM LoRA adapters.
3. After 5 steps, ``trainer.save_model()`` writes the pipeline directory
   layout under ``OUTPUT_DIR``.

Status: **deferred** to a node with GPU access. The unit test
``test_compute_gen_loss_finite_with_grad`` exercises the same forward path
on CPU using stand-in modules.

## 4. Save → reload roundtrip (new pipeline output)

Procedure (after a smoke run has produced ``OUTPUT_DIR``):

```python
import torch, sys
sys.path.insert(0, '/scratch/network/ssd2/junlin/ssl_mllm/InternVL-U')

from internvl.model.internvlu import InternVLUUnifiedModel
m = InternVLUUnifiedModel.from_pretrained_components(
    OUTPUT_DIR, vlm=..., torch_dtype=torch.bfloat16,
)
print('reloaded as InternVLUUnifiedModel OK')

from internvlu import InternVLUPipeline
p = InternVLUPipeline.from_pretrained(OUTPUT_DIR, torch_dtype=torch.bfloat16)
print('reloaded as InternVLUPipeline OK')
```

Status: **CPU-side roundtrip verified** by the unit test
``test_save_then_reload_roundtrip``, which checks that
``save_pretrained`` writes weight files for every subcomponent and that a
state_dict load round-trips. The full ``InternVLUPipeline.from_pretrained``
check is deferred to GPU.

## 5. Open follow-ups / blockers

None blocking. Soft follow-ups:

* **Anyres mode** (``config.anyres_image_size = True``) is left disabled
  in the new full-unified recipe — supporting it would require feeding
  ``image_grid_thw`` through the conditioning path.
* **Packed datasets** (``use_packed_ds=True``) are disabled in the new
  unified path; the existing VLM packed path is unchanged.
* **Image-editing CFG** (``conditional_image``) is supported by the
  inference pipeline but not yet wired into the imgen dataset/collator.
  See ``ImgenLazyDataset`` for the extension point.

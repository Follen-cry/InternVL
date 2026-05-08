# InternVL-U full unified-SFT — Phase 0 design notes

These notes capture everything required to extend the existing VLM-only LoRA SFT
into a full pipeline SFT (VLM + generation_decoder + optional VAE) without
breaking the existing flow.

## 1. Snapshot layout

The base checkpoint is the HuggingFace pipeline snapshot at:

```
/homes/55/junlin/.cache/huggingface/hub/models--InternVL-U--InternVL-U/snapshots/f012d760e69712bb47f7d3d09a24280f346cee01
```

`model_index.json` (verbatim):

```json
{
    "_class_name": "InternVLUPipeline",
    "vlm":                ["internvlu.vlm",                       "InternVLUChatModel"],
    "generation_decoder": ["internvlu.diffusion",                 "InternVLUGenerationDecoder"],
    "vae":                ["diffusers",                           "AutoencoderKLQwenImage"],
    "scheduler":          ["diffusers",                           "DPMSolverMultistepScheduler"],
    "processor":          ["internvlu.processing_internvlu",      "InternVLUProcessor"]
}
```

| Component            | Class                                  | Library      | Has trainable params? | Default plan |
|----------------------|----------------------------------------|--------------|-----------------------|---------------|
| `vlm`                | `InternVLUChatModel`                   | local pkg    | yes                   | LoRA / freeze parts (existing recipe) |
| `generation_decoder` | `InternVLUGenerationDecoder`           | local pkg    | yes (transformer + projector + encoder_padding_token) | trainable in new recipe |
| `vae`                | `AutoencoderKLQwenImage`               | diffusers    | yes (we choose to freeze) | frozen (default) |
| `scheduler`          | `DPMSolverMultistepScheduler`          | diffusers    | no (config only)      | frozen, used at inference; training uses its own flow-matching schedule (see §4) |
| `processor`          | `InternVLUProcessor`                   | local pkg    | no                    | frozen |

The `internvlu` Python package source (used by `from internvlu import InternVLUPipeline`)
lives at `/scratch/network/ssd2/junlin/ssl_mllm/InternVL-U/internvlu/`. It is **not**
pip-installed; downstream tools either import via that path or treat the snapshot
subdirs as `trust_remote_code=True` payloads (each subdir's `config.json` carries
an `auto_map`).

## 2. Conditioning contract (VLM → generation_decoder)

Source of truth: `internvlu/pipeline_internvlu.py::_prepare_diffusion_inputs`
(lines 339–428) and `_prepare_hidden_state_mask` (141–263). What follows is an
exact transcription, with annotations for the training-time forward.

### 2.1 Inputs the VLM produces

`InternVLUChatModel.generate_hidden_states(...)` (file: `internvlu/vlm/modeling_internvlu_chat.py`, lines 538–596)
runs the language model with `output_hidden_states=True` and `padding_type="pad"`,
returning a `CausalLMOutputWithPast` whose `hidden_states` is a tuple of length
`num_hidden_layers + 1`.

For training we cannot use that method directly because it is decorated with
`@torch.no_grad`. Instead, the unified model replays the same body **without**
the no-grad wrapper. The body, exactly:

1. `input_embeds = lm.get_input_embeddings()(input_ids)`
2. `input_embeds = self.replace_img_special_tokens(input_embeds, input_ids)`
3. If `pixel_values is not None`: extract ViT features via `self.extract_feature(pixel_values)` and scatter into `input_embeds` at positions where `input_ids == self.img_context_token_id`.
4. `outputs = self.language_model(inputs_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=True, return_dict=True, padding_type="pad")`

### 2.2 Hidden-state selection

```python
# pipeline_internvlu.py, _prepare_diffusion_inputs
vlm_hidden_states = [
    vlm_hidden_states[i].view(B, L, -1)
    for i in self.generation_decoder.config.vlm_select_layer
]
vlm_hidden_states = torch.cat(vlm_hidden_states, dim=-1)  # B, N, C * num_layers
```

For the released checkpoint `vlm_select_layer = [-1, -2]` and the LLM hidden
size is 2048; concatenated dim is 4096, matching `input_hidden_size` in the
generation_decoder config.

### 2.3 Token-position mask

`_prepare_hidden_state_mask(input_ids, attention_mask, generation_flags, padding_type="pad")`
selects, for **each** image-to-generate `k`, the contiguous span of tokens from
the second `<|im_start|>` token within the same row up to (and including) the
`<img>` start token. Code excerpt:

```python
img_start_positions = (input_ids == self.vlm.img_start_token_id).nonzero()
gen_img_start_positions = img_start_positions[generation_flags.bool()]  # [K, 2]
state_positions_row = torch.arange(B, device=...)[None]
state_positions_col = torch.arange(N, device=...)[None]
state_mask_row = state_positions_row == gen_img_start_positions[:, :1]   # [K, B]
state_mask_col = state_positions_col <= gen_img_start_positions[:, 1:]   # [K, N]

# (pad branch; we only use this for training)
im_start_second_positions_to_gen_img_start = ...   # second <|im_start|> in row
state_mask_col = state_mask_col & (
    im_start_second_positions_to_gen_img_start[:, 1:] <= state_positions_col
)
state_mask = (state_mask_row[..., None] & state_mask_col[:, None]).bool() & attention_mask[None]
```

The result is a list of `K` per-image tensors, each shape
`(seq_len_k, hidden_size)` = `(seq_len_k, 4096)`, fed as `encoder_hidden_states`
to the generation_decoder.

### 2.4 `encoder_image_token_mask`

```python
selected = input_ids == self.vlm.img_context_token_id
vlm_image_token_mask = [selected[s] for s in state_mask]
```

A boolean mask, same shape as each `encoder_hidden_states[k]`, marking the
positions inside the conditioning span that hold ViT tokens (used by the
decoder's joint attention to know which tokens are textual vs visual).

### 2.5 Decoder projection / normalization

`InternVLUGenerationDecoder.prepare_forward_input` (file:
`internvlu/diffusion/modeling_internvlu_generation_decoder.py`, lines 75–153)
performs three steps over the variable-length list:

1. Pad to `max_sequence_length` (or longest + max_seq) using `encoder_padding_token` (a learnable parameter).
2. Build a boolean attention mask `(B, max_len)`.
3. Apply `self.decoder_projector` — for the released config this is `nn.Identity()` (`decoder_projector_type = "identity"`), so it is a *no-op* on the bytes but the projector is still present in the state_dict.

The decoder's `txt_norm` and `txt_in` **are run inside the transformer's
`forward`**, before any layer (`internvl_transformer.py` line 2673–2674):

```python
encoder_hidden_states = self.txt_norm(encoder_hidden_states)
encoder_hidden_states = self.txt_in(encoder_hidden_states)
```

The training forward must therefore **not** apply txt_norm/txt_in itself; the
decoder forward does it.

## 3. Image / latent contract

Inference path (`pipeline_internvlu_generation_decoder.py::pixels_to_latents`,
lines 251–279):

```python
x = x.unsqueeze(2)                     # [B, C, 1, H, W]  (Qwen-Image VAE expects T axis)
image_latents = retrieve_latents(self.vae.encode(x_bs), generator=None, sample_mode="argmax")
latents_mean = vae.config.latents_mean  # length 16
latents_std  = vae.config.latents_std   # length 16
z = (image_latents - latents_mean) / latents_std
z = z.squeeze(2)                       # [B, 16, H', W']
```

For training we use `sample_mode="sample"` (standard VAE-encoder use) and
disable grads for the VAE (it's frozen by default).

## 4. Training-time noise / loss

The pipeline ships with a `DPMSolverMultistepScheduler` configured for
**inference** (flow_prediction, dpmsolver++, flow_shift=3.0). The
`InternVLUGenerationDecoderConfig` supplies the **training** schedule:

```
weighting_scheme = "logit_normal"
logit_mean       = 0.0
logit_std        = 1.0
mode_scale       = 1.29
flow_shift       = 3.0
```

Standard SD3-style flow-matching recipe (verified against `diffusers`
training scripts, e.g. `examples/dreambooth/train_dreambooth_sd3.py`):

1. `u ~ Normal(logit_mean, logit_std)`; then `sigma = sigmoid(u)` ∈ (0, 1).
2. Apply `flow_shift`: `sigma_shifted = (flow_shift * sigma) / (1 + (flow_shift - 1) * sigma)`.
3. Build noisy latent: `noisy = (1 - sigma_shifted) * z + sigma_shifted * noise`.
4. Target (flow_prediction velocity): `target = noise - z`.
5. Use `timestep = sigma_shifted * 1000` (int) when calling the decoder.
6. `loss = MSE(pred, target)` (no per-pixel weighting; `mask_weight_type=null`,
   `sigmas_as_weight=false`, `region_weighting=false`).

The new code uses these utilities directly. `flow_shift` is read from the
generation_decoder config so future checkpoints with a different shift remain
in-distribution.

## 5. CFG dropout (for `cfg_dropout` flag)

At inference the CFG triplet is `(full_cond, part_cond, uncond)`. At training,
`<img_uncond>` token replaces the conditioning prompt with probability
`cfg_dropout` (default 0.1). The unconditional embedding is produced by
swapping the entire conditioning token range to `IMG_UNCOND_TOKEN` before
embedding lookup, so `replace_img_special_tokens` substitutes the learned
`special_token_embedding[IMG_UNCOND_TOKEN]`.

## 6. Save / load contract

`InternVLUUnifiedModel.save_pretrained(out)` writes:

```
out/
├── model_index.json           # carries the same _class_name = InternVLUPipeline manifest
├── vlm/                       # produced by InternVLUChatModel.save_pretrained
├── generation_decoder/        # produced by InternVLUGenerationDecoder.save_pretrained
├── vae/                       # AutoencoderKLQwenImage.save_pretrained
├── scheduler/                 # DPMSolverMultistepScheduler.save_pretrained (config only)
└── processor/                 # InternVLUProcessor.save_pretrained
```

The output is directly loadable by `InternVLUPipeline.from_pretrained(out)`,
**provided** the `internvlu` package is on the Python path. Verified by Phase 6
test #3.

## 7. Phase-0 baseline marker

We do **not** run a 1-step real-model baseline as part of Phase 0 because that
requires GPUs + several minutes of weight loading. Instead we record the
*structural* baseline that Phase 6 must reproduce bit-identically:

- Trainable-parameter list for `shell/internvlu/internvlu_4b_sft_lora.sh`:
  - everything LoRA emitted by `model.wrap_llm_lora(r=16, lora_alpha=32)` on `language_model`
  - all parameters of `model.mlp1`
  - all parameters of `model.special_token_embedding`
- All other parameters (`vision_model.*`, the original LLM weights, etc.) frozen.

The Phase 6 backward-compat test checks:

1. `internvl_chat_finetune_u.py` (unmodified) on `synth_meta_understanding.json`
   produces the same trainable-name set as a recorded gold list, AND
2. step-1 training loss matches a recorded gold value within 1e-4 (deterministic
   seed pinned via `--seed`).

The gold values are produced by running the existing script once with
`OUTPUT_DIR=/tmp/baseline-gold` and the synthetic meta. The driver lives at
`internvl_chat/tests/run_baseline.sh` and is documented in `UNIFIED_VERIFICATION.md`.

## 8. Caveats and assumptions

* **Anyres mode** (`config.anyres_image_size = True`) is not exercised in the
  existing VLM-only SFT and is similarly disabled in the new full-SFT recipe.
  Supporting it would require also passing `image_grid_thw` through the
  conditioning path; left as a follow-up.
* **CFG triplet sequencing.** The inference pipeline expects encoder hidden
  states to be ordered as `[full_cond_batch, part_cond_batch, uncond_batch]`
  (see the `assert len(encoder_hidden_states) % 3 == 0` in
  `pipeline_internvlu_generation_decoder.py`). At training we feed the decoder
  exactly the prepared list **without** triplication; the decoder's CFG
  triplet path is inference-only and gated by the pipeline call (see
  `latent_model_input = torch.cat([latents] * 3)`). We confirmed by reading
  `InternVLUTransformer2DModel.forward` (line 2614) that there is no CFG-aware
  branching inside the decoder itself — it is a plain transformer that
  consumes `(hidden_states, encoder_hidden_states, encoder_attention_mask, …)`.
* **Padding type.** Training uses `padding_type="pad"`. Packed
  (`use_packed_ds=True`) is left out of the new full-unified recipe for now to
  keep the conditioning-mask logic simple; the existing VLM packed path is
  unchanged.

## 9. Files referenced

```
internvl_chat/internvl/model/internvlu/
internvl_chat/internvl/train/internvl_chat_finetune_u.py
internvl_chat/shell/internvlu/internvlu_4b_sft_lora.sh
internvl_chat/shell/internvlu/internvlu_4b_sft_full.sh
internvl_chat/tools/assemble_unified.py
internvl_chat/tools/merge_lora_u.py
/scratch/network/ssd2/junlin/ssl_mllm/InternVL-U/internvlu/pipeline_internvlu.py
/scratch/network/ssd2/junlin/ssl_mllm/InternVL-U/internvlu/diffusion/modeling_internvlu_generation_decoder.py
/scratch/network/ssd2/junlin/ssl_mllm/InternVL-U/internvlu/diffusion/internvlu_transformer.py
/scratch/network/ssd2/junlin/ssl_mllm/InternVL-U/internvlu/diffusion/pipeline_internvlu_generation_decoder.py
/scratch/network/ssd2/junlin/ssl_mllm/InternVL-U/internvlu/diffusion/configuration_internvlu_generation_decoder.py
/scratch/network/ssd2/junlin/ssl_mllm/InternVL-U/internvlu/vlm/modeling_internvlu_chat.py
```

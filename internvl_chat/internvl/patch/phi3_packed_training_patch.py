# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Phi3 packed training patch disabled: PHI3_ATTENTION_CLASSES and
# Phi3FlashAttention2 were removed in transformers >= 4.44.
# Not applicable for InternVL-U (Qwen2.5 backbone).

def replace_phi3_attention_class():
    print('Skipping replace_phi3_attention_class: '
          'not applicable for Qwen2.5 backbone with transformers >= 4.44.')

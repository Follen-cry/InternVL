# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# QWEN2_ATTENTION_CLASSES and Qwen2FlashAttention2 were removed in
# transformers >= 4.44. The packed training patch is not compatible with
# transformers 4.52.x which uses a unified attention dispatch system.
# group_by_length=True (used instead) handles sequence length batching
# without needing this patch.

def replace_qwen2_attention_class():
    print('Skipping replace_qwen2_attention_class: '
          'QWEN2_ATTENTION_CLASSES removed in transformers >= 4.44.')

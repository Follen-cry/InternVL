# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# LlamaFlashAttention2 and LLAMA_ATTENTION_CLASSES were removed in
# transformers >= 4.44. This patch is only relevant for LLaMA-based models;
# since InternVL-U uses a Qwen2.5 backbone, it is safely disabled here.

def replace_llama_attention_class():
    print('Skipping replace_llama_attention_class: '
          'not applicable for Qwen2.5 backbone with transformers >= 4.44.')

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch
from internvl.model.internvlu import InternVLUChatModel
from transformers import AutoTokenizer
'''
Usage:
python merge_lora_u.py \
  /scratch/network/ssd2/junlin/models/internvlu/internvl-u-4epoch/ \
  /scratch/network/ssd2/junlin/models/internvlu/internvl-u-4epoch-merged/
'''
ap = argparse.ArgumentParser()
ap.add_argument('input_path')
ap.add_argument('output_path')
args = ap.parse_args()

print('Loading model...')
model = InternVLUChatModel.from_pretrained(
    args.input_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
).eval()
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(args.input_path, trust_remote_code=False)

if model.config.use_backbone_lora:
    model.vision_model.merge_and_unload()
    model.vision_model = model.vision_model.model
    model.config.use_backbone_lora = 0
if model.config.use_llm_lora:
    model.language_model.merge_and_unload()
    model.language_model = model.language_model.model
    model.config.use_llm_lora = 0

print('Saving merged model...')
model.save_pretrained(args.output_path)
tokenizer.save_pretrained(args.output_path)
print('Done.')

import os, sys
now_dir = os.getcwd()
sys.path.insert(0, now_dir)
from text.g2pw import G2PWPinyin
g2pw = G2PWPinyin(model_dir="OG_AI_LAB/text/G2PWModel",model_source="OG_AI_LAB/pretrained_models/chinese-roberta-wwm-ext-large",v_to_u=False, neutral_tone_with_five=True)
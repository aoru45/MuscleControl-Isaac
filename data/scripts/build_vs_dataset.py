from tqdm import tqdm
import torch
import joblib
import os
import time
import numpy as np
import json
import yaml
with open("data/motionfix/amt_motionfix_latest.json") as f:
    ori_data = json.load(f)
with open("data/yaml_files/motions_motionfix.yaml") as f:
    motionfix_data = yaml.safe_load(f)
#for motion_id in 
with open("data/motionfix/splits.json") as f:
    splits = json.load(f)
    NAME2SPLIT = {}
    for phase in splits:
        for name in splits[phase]:
            NAME2SPLIT[name] = phase

source_motion_id2idx = {}
target_motion_id2idx = {}

for motion in motionfix_data["motions"]:
    if "source" in motion["file"]:
        _motion_id = os.path.basename(motion["file"])[:-4]
        source_motion_id2idx[_motion_id] = motion["idx"]
    elif "target" in motion["file"]:
        _motion_id = os.path.basename(motion["file"])[:-4]
        target_motion_id2idx[_motion_id] = motion["idx"]
    else:
        raise ValueError(f"Unknown motion type: {motion}")
# TOKENIZER
from transformers import AutoTokenizer, XCLIPTextModel
model = XCLIPTextModel.from_pretrained("microsoft/xclip-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")

new_dataset = {
    phase: [] for phase in splits.keys()
}
for motion_id in tqdm(source_motion_id2idx.keys()):
    start = time.time()
    motion_anno = ori_data[motion_id]["annotation"]
    motion_sim = ori_data[motion_id]["similarity_score"]
    with torch.inference_mode():
        inputs = tokenizer(
            motion_anno,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        outputs = model(**inputs)
        pooled_output = outputs.pooler_output  # pooled (EOS token) states
    for it in range(1):
        src_data = dict(np.load(f"outputs/rollouts_pulse/motion_{source_motion_id2idx[motion_id]:06d}_{it:02d}.npz"))
        tgt_data = dict(np.load(f"outputs/rollouts_pulse/motion_{target_motion_id2idx[motion_id]:06d}_{it:02d}.npz"))
        _data = {
            "src": {k:v for k, v in src_data.items() if k != "mus"},
            "tgt": {k:v for k, v in tgt_data.items() if k != "mus"},
            "anno": ori_data[motion_id],
            "embed": pooled_output, # (1, 512)
        }
        phase = NAME2SPLIT[motion_id]
        new_dataset[phase].append(_data)
for phase in new_dataset.keys():
    joblib.dump(new_dataset[phase], f"data/motionfix/motionfix_vs_pulse_{phase}.pth")



    

import numpy as np
import time
import os
import xml.etree.ElementTree as ET
import json
import yaml
import matplotlib.pyplot as plt
from google import genai

def _ema(seq, alpha=0.2):
    x = np.asarray(seq, dtype=np.float64)
    y = np.empty_like(x)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = alpha * x[t] + (1.0 - alpha) * y[t - 1]
    return y
class BiomechanicalTemporalSummarizer:
    def __init__(self, xml_path):
        self.xml_path = xml_path
        
        # 1. 定义解剖学大组 (Group)
        self.group_definition = {
            "Core": ["Torso", "Spine", "Neck"],
            "Hips": ["Pelvis"],
            "Thighs": ["FemurL", "FemurR"],
            "Calves": ["TibiaL", "TibiaR"],
            "Shoulders": ["ShoulderL", "ShoulderR"],
            "Arms": ["ArmL", "ArmR", "ForeArmL", "ForeArmR"]
        }
        self.idx_to_group = self._build_mapping()
        self.group_names = list(self.group_definition.keys())

    def _build_mapping(self):
        """解析 XML，将 284 个索引归类到 G 个大组"""
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        mapping = {}
        for i, unit in enumerate(root.findall('Unit')):
            body = unit.find('Waypoint').get('body')
            for group_name, keywords in self.group_definition.items():
                if any(kw in body for kw in keywords):
                    mapping[i] = group_name
                    break
        return mapping

    def _process_sequence(self, act_data, indices):
        group_seq = np.max(act_data[:, indices], axis=1)
        return _ema(group_seq).tolist()

    def get_gpt_prompt(self, entry_id, json_data, act_src, act_tgt):
        """生成包含时序向量的 Prompt"""
        biomech_report = {}
        
        for g_name in self.group_names:
            indices = [idx for idx, name in self.idx_to_group.items() if name == g_name]
            if not indices: continue
            
            biomech_report[g_name] = {
                "src_timeline": self._process_sequence(act_src, indices),
                "tgt_timeline": self._process_sequence(act_tgt, indices)
            }

        # 构造最终给 GPT 的输入
        prompt_input = {
            "id": entry_id,
            "original_annotation": json_data['annotation'],
            "motion_similarity": json_data['similarity_score'],
            "biomechanical_timelines": biomech_report
        }
        s = json.dumps(prompt_input, indent=2, ensure_ascii=False)

        system_prompt = f"""
你是一位生物力学教练。请对比 Source 和 Target 的肌肉激活时序向量（每个向量代表动作从开始到结束的发力曲线）。

任务：为数据集添加 "annotation_null" 字段。
要求：
1. 观察 timeline 中的数值变化：
   - 数值大小代表发力强度 (Effort)。
   - 数值在序列中的位置代表发力时机 (Timing/Phase)。
   - 多个部位同时高数值代表刚度 (Stiffness/Co-contraction)。
2. 风格参考 MotionFix：极简、行动导向。
3. 必须强调：保持轨迹 "{json_data['annotation']}" 不变，仅改变执行风格。

数据如下：
{s}

请直接返回补全了 "annotation_null" 的 JSON 对象, 只包含annotation_null字段，描述用英文,要求参考annotation一样简短直接。尽量只描述单一部位。不要重复描述annotation字段，只补充肌肉激活描述。
"""
        return system_prompt

if __name__ == "__main__":
    # --- 使用示例 ---
    summarizer = BiomechanicalTemporalSummarizer("protomotions/data/assets/muscle284.xml")



    with open("data/motionfix/amt_motionfix_latest.json") as f:
        ori_data = json.load(f)
    with open("data/yaml_files/motions_motionfix.yaml") as f:
        motionfix_data = yaml.safe_load(f)
    #for motion_id in 


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
    client = genai.Client(api_key="AIzaSyBPGCfQTtOP6peuvI0Yt2si5nyrnk593d4")
    for motion_id in source_motion_id2idx.keys():
        start = time.time()
        motion_anno = ori_data[motion_id]["annotation"]
        motion_sim = ori_data[motion_id]["similarity_score"]

        act_src = np.load(f"outputs/rollouts_sa/motion_{source_motion_id2idx[motion_id]:06d}.npz")["activations"]
        act_tgt = np.load(f"outputs/rollouts_sa/motion_{target_motion_id2idx[motion_id]:06d}.npz")["activations"]


        

        # # 生成 Prompt
        prompt = summarizer.get_gpt_prompt(motion_id, {"annotation": motion_anno, "similarity_score": motion_sim}, act_src, act_tgt)
        print(prompt)

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
        )
        end = time.time()

        print(response.text)
        fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
        axes = axes.flatten()

        for i, g_name in enumerate(summarizer.group_names):
            indices = [idx for idx, name in summarizer.idx_to_group.items() if name == g_name]
            if not indices:
                continue
            src_seq = summarizer._process_sequence(act_src, indices)
            tgt_seq = summarizer._process_sequence(act_tgt, indices)
            ax = axes[i]
            ax.plot(src_seq, label="source")
            ax.plot(tgt_seq, label="target")
            ax.set_title(g_name)
            ax.legend(loc="upper right")
        for j in range(len(summarizer.group_names), len(axes)):
            fig.delaxes(axes[j])
        fig.suptitle("Activation Group Timelines")
        plt.tight_layout()
        plt.show()

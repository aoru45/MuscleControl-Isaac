import os
import sys
import yaml
import numpy as np


def collect_motion_entries(motions_root: str, yaml_dir: str):
    entries = []
    idx = 0
    for root, _, files in os.walk(motions_root):
        for f in sorted(files):
            if not f.endswith('.npy'):
                continue
            abs_path = os.path.join(root, f)
            try:
                d = np.load(abs_path, allow_pickle=True).item()
                fps = float(d.get('fps', 30.0))
            except Exception:
                fps = 30.0
            rel_to_yaml = os.path.relpath(abs_path, start=yaml_dir)
            entry = {
                'file': rel_to_yaml,
                'fps': fps,
                'idx': idx,
                'sub_motions': [
                    {
                        'idx': idx,
                        'weight': 1.0,
                    }
                ]
            }
            entries.append(entry)
            idx += 1
    return entries


def main():
    motions_root = sys.argv[1] if len(sys.argv) > 1 else 'data/motions_retarget_filtered'
    output_yaml = sys.argv[2] if len(sys.argv) > 2 else 'data/yaml_files/motions_retarget_filtered.yaml'

    yaml_dir = os.path.dirname(os.path.abspath(output_yaml))
    os.makedirs(yaml_dir, exist_ok=True)

    entries = collect_motion_entries(motions_root, yaml_dir)
    doc = {'motions': entries}

    with open(output_yaml, 'w') as f:
        yaml.safe_dump(doc, f, sort_keys=False)

    print(f"Wrote {len(entries)} motions to {output_yaml}")


if __name__ == '__main__':
    main()


import os
import json
import re
from argparse import ArgumentParser
from util import HistoryCategory

parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--model_inference_dirs', type=str)
parser.add_argument('--history_settings', type=str)
args = parser.parse_args()

CURRENT_DIR_PATH = os.path.abspath(os.path.dirname(__file__))

for model_inference_dir in args.model_inference_dirs.split(','):
    for history_flag in args.history_settings.split(','):
        name_suf = f"{history_flag}"
        _evaluate_path = f"{CURRENT_DIR_PATH}/{args.dataset}/{model_inference_dir}"
        if not os.path.exists(_evaluate_path):
            print(f'error: _evaluate_path {_evaluate_path} not exist!!')
        merged_evaluate_path = os.path.join(_evaluate_path, f"unittest_result_{HistoryCategory(history_flag).name}.json")
        if os.path.exists(merged_evaluate_path):
            os.remove(merged_evaluate_path)
            print(f"Removed existing {merged_evaluate_path}")

        merged_data = {}

        pattern = re.compile(rf"unittest_result_{HistoryCategory(history_flag).name}_(\d+)\.json$")
        all_files = os.listdir(_evaluate_path)
        all_bugs_eval_files = [
            os.path.join(_evaluate_path, f)
            for f in all_files if pattern.match(f)
        ]
        for each_bug_eval_file in all_bugs_eval_files:
            with open(each_bug_eval_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                merged_data.update(data)

        with open(merged_evaluate_path, 'w', encoding='utf-8') as f:
            json.dump(dict(sorted(merged_data.items(), key=lambda x: int(x[0]))), f, indent=2)

        print(f"Merged {len(all_bugs_eval_files)} files into {merged_evaluate_path}")

        # Delete the individual per-bug files
        for each_bug_eval_file in all_bugs_eval_files:
            os.remove(each_bug_eval_file)
        print(f"Deleted {len(all_bugs_eval_files)} individual files for setting {history_flag}")


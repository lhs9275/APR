import json
import os
import glob
import numpy as np
from argparse import ArgumentParser
from scipy.special import comb # 💡 표준 조합(comb) 계산을 위해 추가

# util.py에 initialize_result_dict가 있다고 가정합니다. 만약 없다면 이 라인은 제거해야 합니다.
# from util import initialize_result_dict

CURRENT_DIR_PATH = os.path.abspath(os.path.dirname(__file__))


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--dataset', default="bugsinpy", type=str)
    parser.add_argument('--model_inference_dirs', default="patch", type=str)
    parser.add_argument('--k_list', type=str, default="1,2,3,4,5,6,7,8,9,10",
                        help="Comma-separated list of k values, e.g., '1,3,5,10'")
    return parser


def main():
    args = get_parser().parse_args()
    dataset_name = args.dataset
    k_list = [int(k) for k in args.k_list.split(',')]

    for model_inference_dir in args.model_inference_dirs.split(','):
        print(f'Evaluating pass_at_k result for {dataset_name} on {model_inference_dir}')

        _evaluate_path = f"{CURRENT_DIR_PATH}/{dataset_name}/{model_inference_dir}"
        if not os.path.exists(_evaluate_path):
            raise ValueError(f'error: _evaluate_path {_evaluate_path} not exist!!')

        file_pattern = f"{_evaluate_path}/unittest_result_5.PatchesResults10.json"
        eval_files = glob.glob(file_pattern)

        if not eval_files:
            print(f"Warning: No evaluation files found matching pattern: {file_pattern}\n")
            continue

        print(f"Found {len(eval_files)} files to process: {eval_files}")

        combined_eval_results = {}
        for file_path in eval_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    combined_eval_results.update(data)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Warning: Could not read or parse {file_path}. Skipping. Error: {e}")

        # <--- 수정된 부분: total_bug_count=51 인자를 전달합니다.
        result_ = _evaluate_average_pass_at_k(combined_eval_results, k_list, total_bug_count=51)

        if result_ is None:
            print(f'Could not calculate pass@k for {model_inference_dir}, skipping.\n')
            continue

        print(f'Result: {result_}\n')

        pass_k_result_path = f"{_evaluate_path}/pass_k_result_5.PatchesResults10_.txt"
        with open(pass_k_result_path, 'a') as pass_k_result:
            pass_k_result.write(f'pass_at_k_final: {result_}\n')


def _evaluate_average_pass_at_k(eval_result_json, k_list, total_bug_count=None):
    """
    total_bug_count: 전체 벤치마크의 버그 수를 지정합니다.
                     None이면 JSON에 있는 버그 수만 사용합니다. (기존 방식)
    """
    if not eval_result_json:
        print("Error: The provided evaluation data is empty.")
        return None

    pass_at_k_result = {}

    scores_per_bug = []
    for bug_id, eval_result in eval_result_json.items():
        if 'nucleus_sampling_flags' not in eval_result:
            continue

        flags = eval_result['nucleus_sampling_flags']
        n = len(flags)
        c = flags.count('Pass')

        if n == 0:
            continue

        scores_per_bug.append({'n': n, 'c': c})

    if not scores_per_bug:
        print("Warning: No valid bug results found to calculate pass@k.")
        return {}

    # --- 로직 수정 부분 시작 ---

    # 평가의 기준이 될 전체 버그 수를 결정합니다.
    # 인자로 받은 total_bug_count가 있으면 그 값을 사용하고, 없으면 기존 방식대로 처리된 버그 수를 사용합니다.
    processed_bugs_count = len(scores_per_bug)
    bugs_to_evaluate_on = total_bug_count if total_bug_count is not None else processed_bugs_count

    # 성공한 버그 수는 'c'가 0보다 큰 버그의 개수입니다. 이 로직은 동일합니다.
    solved_bugs = sum(1 for item in scores_per_bug if item['c'] > 0)

    for k in sorted(k_list):
        # 처리된 버그들의 pass@k 점수의 '총합'을 먼저 계산합니다.
        # 결과가 없는 버그는 점수가 0이므로 합산에 영향을 주지 않습니다.
        total_pass_at_k_score = sum(_compute_pass_at_k(item['n'], item['c'], k) for item in scores_per_bug)

        # '총합'을 전체 기준 버그 수(예: 51)로 나누어 올바른 평균을 계산합니다.
        pass_at_k = total_pass_at_k_score / bugs_to_evaluate_on

        # 출력 형식에서 분모를 올바른 기준 버그 수로 변경합니다.
        pass_at_k_result[f"pass@{k}"] = f"{pass_at_k:.2%} ({solved_bugs}/{bugs_to_evaluate_on})"

    # --- 로직 수정 부분 끝 ---

    return pass_at_k_result


def _compute_pass_at_k(n, c, k):
    """
    pass@k를 표준 조합(combination) 공식을 사용하여 정확하게 계산합니다.
    
    n: 전체 샘플 수
    c: 성공(통과) 샘플 수
    k: 선택할 샘플 수
    """
    # 💡 [수정 1] 성공 샘플이 하나도 없다면(c=0), 확률은 항상 0입니다.
    if c == 0:
        return 0.0

    # 💡 [수정 2] 실패 샘플 수(n-c)가 뽑으려는 수(k)보다 적으면, 반드시 성공 샘플을 포함하게 되므로 확률은 1입니다.
    if n - c < k:
        return 1.0

    # 💡 [수정 3] 표준 조합 공식을 사용하여 정확한 확률을 계산합니다.
    # pass@k = 1 - (k개의 실패 샘플만 뽑을 확률)
    return 1.0 - (comb(n - c, k) / comb(n, k))


if __name__ == "__main__":
    main()
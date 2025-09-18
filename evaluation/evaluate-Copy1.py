import json
import os
import subprocess
import textwrap
import logging # 1. 로깅 모듈 임포트
from argparse import ArgumentParser
from datetime import datetime
from dataset_adapter import BugsInPy, Defects4J, DatasetAdapter

CURRENT_DIR_PATH = os.path.abspath(os.path.dirname(__file__))
PROJECT_DIR_BASE = os.path.abspath(os.path.join(CURRENT_DIR_PATH, '../'))
MODEL_INFERENCE_BASE_PATH = os.path.abspath(os.path.join(PROJECT_DIR_BASE, 'Results'))


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--dataset', default="bugsinpy", type=str)
    parser.add_argument('--model_inference_dirs', default="patch", type=str)
    parser.add_argument('--history_settings', default='1', type=str)
    parser.add_argument('--bug_id_list', type=str, default='')
    return parser


def adapter_factory(dataset_name):
    if dataset_name == "bugsinpy":
        return BugsInPy()
    elif dataset_name == "defects4j":
        return Defects4J()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def setup_logging():
    """로깅 기본 설정을 수행하는 함수"""
    logging.basicConfig(
        level=logging.INFO, # INFO 레벨 이상의 로그를 모두 기록
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("evaluation.log", mode='w', encoding='utf-8'), # 파일에 로그 기록
            logging.StreamHandler() # 콘솔에 로그 출력
        ]
    )

def main():
    # 2. 로깅 설정 실행
    setup_logging()

    args = get_parser().parse_args()
    adapter = adapter_factory(args.dataset)

    bug_filter = set(args.bug_id_list.split(',')) if args.bug_id_list else None

    evaluate_dir_to_file_dict = {}
    for model_inference_dir in args.model_inference_dirs.split(','):
        model_inference_dir = model_inference_dir.strip()
        if not model_inference_dir:
            continue
        evaluate_dir_to_file_dict[model_inference_dir] = {}

        def resolve_json_path(base_entry: str) -> str:
            if base_entry.lower().endswith('.json') and os.path.isfile(base_entry):
                return base_entry
            candidate = os.path.join(MODEL_INFERENCE_BASE_PATH, base_entry, "5.PatchesResults10.json")
            return candidate

        for _ in args.history_settings.split(','):
            path = resolve_json_path(model_inference_dir)
            if os.path.exists(path):
                # 3. 기존 print 문을 logging으로 교체
                logging.info(f"Successfully found inference file: {path}")
                with open(path, 'r', encoding='utf-8') as f:
                    evaluate_dir_to_file_dict[model_inference_dir][path] = json.load(f)
            else:
                logging.warning(f"Could not find inference file: {path}")

    logging.info(f"Constructed evaluate_dir_to_file_dict with {len(evaluate_dir_to_file_dict)} entries.")

    bugs_meta_data_file = f"{PROJECT_DIR_BASE}/bugs_meta_data.json"
    bugs_meta_data: dict = json.load(open(bugs_meta_data_file, 'r', encoding='utf-8'))
    ground_error = {}

    for bug_id, bug_meta_data in bugs_meta_data.items():
        if bug_filter and bug_id not in bug_filter:
            continue

        if adapter.should_skip_bug(bug_id):
            logging.info(f"Adapter policy: Skipping bug {bug_id}")
            continue

        bug_id_key = 'bugsinpy_id' if adapter.dataset_name == 'bugsinpy' else 'defects4j_id'
        project_name = adapter.map_project_name(bug_meta_data['project_name'])
        project_checkout_path = adapter.build_project_path(project_name, str(bug_meta_data[bug_id_key]))
        
        logging.info(f"Processing bug: {bug_id} ({project_name}-{bug_meta_data[bug_id_key]})")

        # 1) checkout
        checkout_flag = adapter.checkout(project_name, str(bug_meta_data[bug_id_key]), project_checkout_path)
        if not checkout_flag:
            ground_error[bug_id] = 'Error: checkout'
            logging.error(f"Ground truth checkout failed for {bug_id}. Skipping.")
            continue

        # 2) compile
        project_checkout_path = (
            os.path.join(project_checkout_path, project_name)
            if adapter.dataset_name == 'bugsinpy'
            else project_checkout_path
        )
        compile_flag = adapter.compile(project_checkout_path)
        if not compile_flag:
            ground_error[bug_id] = 'Error: compile'
            logging.error(f"Ground truth compile failed for {bug_id}. Skipping.")
            continue

        # 3) test ground truth
        test_ground_flag = adapter.test(project_checkout_path)
        if test_ground_flag != 'Plausible':
            ground_error[bug_id] = test_ground_flag
            logging.error(f"Ground truth test failed for {bug_id} with status '{test_ground_flag}'. Skipping.")
            continue
        
        logging.info("Ground truth fixed code passed the test. Proceeding to evaluate model-generated patches.")
        logging.info("===========================================================================\n")

        # 4) evaluate model-generated code
        for model_inference_dir, path_file_dict in evaluate_dir_to_file_dict.items():
            _evaluate_path = f"{CURRENT_DIR_PATH}/{adapter.dataset_name}/{sanitize_name(model_inference_dir)}"
            os.makedirs(_evaluate_path, exist_ok=True)
            for path, model_inference_json in path_file_dict.items():

                if bug_id not in model_inference_json:
                    continue
                inference_value = model_inference_json[bug_id]

                choices = inference_value.get('choices', [])
                candidate_codes = [c.get('code') for c in choices if c.get('code')]
                
                if not candidate_codes:
                    code_greedy = (
                        inference_value.get('output', {})
                        .get('greedy_search', {})
                        .get('code')
                    )
                    if code_greedy:
                        candidate_codes.append(code_greedy)

                name_suf = os.path.splitext(os.path.basename(path))[0]
                evaluate_path = f"{_evaluate_path}/unittest_result_{name_suf}_{bug_id}.json"
                pass_k_result_path = f"{_evaluate_path}/pass_k_result_{name_suf}.txt"

                with open(pass_k_result_path, 'a', encoding='utf-8') as pass_k_result:
                    pass_k_result.write("=========================================\n")
                    result_bug_id = {'nucleus_sampling': candidate_codes, 'nucleus_sampling_flags': []}
                    logging.info(f"Start evaluation: {adapter.dataset_name}, {model_inference_dir}, file={name_suf}")

                    for index, nucleus_inference_code in enumerate(result_bug_id['nucleus_sampling']):
                        if not nucleus_inference_code:
                            result_bug_id['nucleus_sampling_flags'].append('Error: empty')
                            continue
                        test_flag_n = execution_tests(adapter, project_checkout_path, bug_meta_data, nucleus_inference_code)
                        logging.info(f'[{index+1}] Testing result for bug {bug_id}: {test_flag_n}')
                        result_bug_id['nucleus_sampling_flags'].append(test_flag_n)

                    log_msg = f"bug id {bug_id} ({project_name}-{bug_meta_data[bug_id_key]}) unittest result: {result_bug_id['nucleus_sampling_flags']}"
                    logging.info(log_msg)
                    pass_k_result.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {log_msg}\n")

                a_new_result = {bug_id: result_bug_id}
                if not os.path.exists(evaluate_path):
                    with open(evaluate_path, 'w', encoding='utf-8') as f:
                        json.dump(a_new_result, f, indent=2)
                else:
                    with open(evaluate_path, 'r+', encoding='utf-8') as f:
                        result = json.load(f)
                        result[bug_id] = result_bug_id
                        f.seek(0)
                        json.dump(result, f, indent=2)
                        f.truncate()

    logging.error(f"Bugs that failed ground truth tests: {ground_error}")
    logging.info("ALL unittests finished!")


def execution_tests(adapter: DatasetAdapter, project_path, bug_meta_data, inference_code) -> str:
    try:
        target_file_path = os.path.join(project_path, bug_meta_data['file']['file_path'])
        target_file_path_backup = target_file_path + '.backup'

        subprocess.run(['cp', target_file_path, target_file_path_backup], check=True)

        function_start = bug_meta_data['function']['function_after_start_line']
        function_end = bug_meta_data['function']['function_after_end_line']

        if adapter.dataset_name == "defects4j":
            function_start, function_end = handle_defects4j_special_cases(
                bug_meta_data, function_start, function_end
            )

        with open(target_file_path, 'r', encoding='utf-8') as f:
            old_file_lines = f.readlines()

        start_line = old_file_lines[function_start - 1]
        start_indent = len(start_line) - len(start_line.lstrip(" "))
        inference_code_indent = adjust_indent(inference_code, start_indent)

        new_file_lines = old_file_lines[:function_start - 1] + [inference_code_indent, '\n'] + old_file_lines[function_end:]
        with open(target_file_path, 'w', encoding='utf-8') as f:
            f.write(''.join(new_file_lines))
    
    except Exception as e:
        logging.error(f"Failed to apply patch before test: {e}", exc_info=True) # exc_info=True는 트레이스백을 함께 기록
        subprocess.run(['mv', target_file_path_backup, target_file_path], check=False)
        return 'Error: before test'

    try:
        test_flag = adapter.test(project_path)
    except Exception as e:
        logging.error(f"Test execution failed: {e}", exc_info=True)
        subprocess.run(['mv', target_file_path_backup, target_file_path], check=False)
        return 'Error: test'

    subprocess.run(['mv', target_file_path_backup, target_file_path], check=False)
    return 'Pass' if test_flag == 'Plausible' else 'Fail'

# setup_evaluation, adjust_indent, handle_defects4j_special_cases, sanitize_name 함수는 변경 없음
# ... (기존 코드와 동일)

def setup_evaluation(adapter: DatasetAdapter, project_name, dataset_bug_id, project_path) -> str:
    # 이 함수는 현재 main 로직에서 직접 사용되지 않지만, 로깅을 추가한다면 아래와 같이 할 수 있습니다.
    try:
        checkout_flag = adapter.checkout(project_name, dataset_bug_id, project_path)
        logging.info(f"Checkout successful for {project_name}-{dataset_bug_id}")
    except Exception as e:
        logging.error(f"Exception during checkout: {e}", exc_info=True)
        return 'Error: checkout'
    if not checkout_flag:
        return 'Error: checkout'

    try:
        compile_flag = adapter.compile(project_path)
        logging.info(f"Compile successful for {project_name}-{dataset_bug_id}")
    except Exception as e:
        logging.error(f"Exception during compile: {e}", exc_info=True)
        return 'Error: compile'
    if not compile_flag:
        return 'Error: compile'
    return "True"


def adjust_indent(code, new_indent):
    dedent_code = textwrap.dedent(code)
    indented_code = textwrap.indent(dedent_code, ' ' * new_indent)
    return indented_code


def handle_defects4j_special_cases(bug_meta_data, default_start, default_end):
    project = bug_meta_data.get("project_name")
    defects4j_bug_id = str(bug_meta_data.get("defects4j_id"))

    if project == "jfreechart":
        special_cases = {
            "1": (1790, 1822), "9": (918, 956), "12": (143, 158),
            "13": (422, 489), "24": (123, 129),
        }
        return special_cases.get(defects4j_bug_id, (default_start, default_end))
    return default_start, default_end


def sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in name)


if __name__ == '__main__':
    main()
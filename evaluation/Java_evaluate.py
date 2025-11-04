import json
import os
import subprocess
import textwrap
import logging
import traceback
import multiprocessing
import shutil
import re
from argparse import ArgumentParser
from datetime import datetime
from dataset_adapter import BugsInPy, Defects4J, DatasetAdapter

# --- 전역 상수 설정 ---
CURRENT_DIR_PATH = os.path.abspath(os.path.dirname(__file__))
PROJECT_DIR_BASE = os.path.abspath(os.path.join(CURRENT_DIR_PATH, '../'))
MODEL_INFERENCE_BASE_PATH = os.path.abspath(os.path.join(PROJECT_DIR_BASE, 'Results'))

# --- Argument Parser ---
def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--dataset', default="defects4j", type=str)
    parser.add_argument('--model_inference_dirs', default="patch", type=str)
    parser.add_argument('--history_settings', default='1', type=str)
    parser.add_argument('--bug_id_list', type=str, default='')
    return parser

# --- 데이터셋 어댑터 팩토리 ---
def adapter_factory(dataset_name):
    if dataset_name == "bugsinpy":
        return BugsInPy()
    elif dataset_name == "defects4j":
        return Defects4J()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

# --- 로깅 설정 ---
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("evaluation_parallel.log", mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def has_command(cmd: str) -> bool:
    return shutil.which(cmd) is not None

# --- 코드 정화/추출 유틸 ---
_CODE_FENCE_OPEN = re.compile(r'^\s*```[a-zA-Z]*\s*$', re.IGNORECASE)
_CODE_FENCE_CLOSE = re.compile(r'^\s*```\s*$', re.IGNORECASE)
_METHOD_LIKE = re.compile(
    r'^\s*(?:@[A-Za-z_][\w.]*(?:\s*\([^)]*\))?\s*)*'   # 어노테이션들
    r'(?:(?:public|protected|private|static|final|synchronized|native|abstract|strictfp)\s+)*'
    r'[\w\<\>\[\]\.?&, \t]+\s+[A-Za-z_]\w*\s*\([^)]*\)\s*\{',  # 반환타입 + 이름(…) {
    re.DOTALL
)

def _strip_code_fences(lines):
    if lines and _CODE_FENCE_OPEN.match(lines[0]):
        lines = lines[1:]
        # 뒤에서 첫 번째 닫는 펜스 제거
        if lines and _CODE_FENCE_CLOSE.match(lines[-1]):
            lines = lines[:-1]
    return lines

def _drop_non_java_noise(lines):
    cleaned = []
    for ln in lines:
        s = ln.rstrip('\n')
        # “Java” / “java”만 있는 줄 제거(앞뒤 공백 허용)
        if s.strip().lower() == 'java':
            continue
        # “...”, “rest of the method/code” 같은 설명 라인 제거
        if re.search(r'\brest of\b', s, re.IGNORECASE) or '...' in s:
            continue
        # 마크다운/설명성 주석으로 보이는 “ # …” 꼬리 텍스트 제거 (Java에는 # 주석 문법이 없음)
        s = re.sub(r'\s+#.*$', '', s)
        cleaned.append(s)
    # 끝 공백 줄 정리
    while cleaned and cleaned[-1].strip() == '':
        cleaned.pop()
    return cleaned

def _extract_body_if_function_decl(s):
    """
    함수 전체가 들어온 경우 { ... } 바디만 추출. 실패하면 원문 반환.
    """
    if not _METHOD_LIKE.match(s):
        return s
    first = s.find('{')
    last = s.rfind('}')
    if first != -1 and last != -1 and last > first:
        inner = s[first+1:last].strip('\n')
        return inner
    return s

def sanitize_inference_code(raw: str) -> str:
    """
    - ```java / ``` 코드펜스 제거
    - 'Java' / 'java' 단독 라인 제거
    - 설명/주석성 잡음 제거
    - 함수 전체가 들어온 경우 함수 바디만 추출
    """
    if not raw:
        return ''
    s = raw.replace('\r\n', '\n').replace('\r', '\n').strip('\n')
    lines = s.split('\n')
    # 펜스 제거
    lines = _strip_code_fences(lines)
    # 잡음 제거
    lines = _drop_non_java_noise(lines)
    s = '\n'.join(lines).strip('\n')
    if not s:
        return ''
    # 함수 바디만 추출 (필요 시)
    s = _extract_body_if_function_decl(s)
    return s.strip('\n')

# --- 멀티프로세싱 워커 함수 ---
def evaluate_bug_worker(args_tuple):
    """
    단일 버그를 평가하는 워커 함수. Pool.map을 위해 단일 인자로 받음.
    """
    bug_id, bug_meta_data, adapter, evaluate_dir_to_file_dict = args_tuple
    process_id = os.getpid()
    
    # ❗ [핵심] 각 프로세스는 독립된 작업 공간을 가집니다.
    unique_checkout_base = os.path.join(PROJECT_DIR_BASE, 'temp_workspaces', f'workspace_{bug_id}_{process_id}')
    
    try:
        os.makedirs(unique_checkout_base, exist_ok=True)
        logging.info(f"Processing bug: {bug_id} in workspace: {unique_checkout_base}")

        bug_id_key = 'bugsinpy_id' if adapter.dataset_name == 'bugsinpy' else 'defects4j_id'
        project_name = adapter.map_project_name(bug_meta_data['project_name'])
        
        # build_project_path는 상대 경로 이름만 생성하도록 원래대로 호출
        relative_project_path = adapter.build_project_path(project_name, str(bug_meta_data[bug_id_key]))
        project_checkout_path = os.path.join(unique_checkout_base, os.path.basename(relative_project_path))

        # 1) checkout
        if not adapter.checkout(project_name, str(bug_meta_data[bug_id_key]), project_checkout_path):
            return bug_id, {'status': 'Error: checkout', 'details': {}}

        # 2) compile
        compile_path = os.path.join(project_checkout_path, project_name) if adapter.dataset_name == 'bugsinpy' else project_checkout_path
        if not adapter.compile(compile_path):
            return bug_id, {'status': 'Error: compile', 'details': {}}

        # 3) test ground truth
        test_ground_flag = adapter.test(compile_path)
        if test_ground_flag != 'Plausible':
            return bug_id, {'status': f'Error: ground_truth_test ({test_ground_flag})', 'details': {}}

        # 4) evaluate model-generated code
        bug_results = {}
        for model_inference_dir, path_file_dict in evaluate_dir_to_file_dict.items():
            for path, model_inference_json in path_file_dict.items():
                if bug_id not in model_inference_json:
                    continue

                inference_value = model_inference_json[bug_id]
                choices = inference_value.get('choices', [])
                candidate_codes = [c.get('code') for c in choices if c.get('code')]
                
                code_greedy = (
                    inference_value.get('output', {})
                    .get('greedy_search', {})
                    .get('code')
                )
                if code_greedy:
                    candidate_codes.append(code_greedy)

                result_bug_id = {'nucleus_sampling': candidate_codes, 'nucleus_sampling_flags': []}
                for index, nucleus_inference_code in enumerate(result_bug_id['nucleus_sampling']):
                    if not nucleus_inference_code:
                        result_bug_id['nucleus_sampling_flags'].append('Error: empty')
                        continue
                    # ⬇️ 주입 전 정화
                    cleaned = sanitize_inference_code(nucleus_inference_code)
                    if not cleaned.strip():
                        result_bug_id['nucleus_sampling_flags'].append('Error: empty-after-sanitize')
                        continue
                    test_flag_n = execution_tests(adapter, compile_path, bug_meta_data, cleaned)
                    result_bug_id['nucleus_sampling_flags'].append(test_flag_n)
                
                bug_results[path] = result_bug_id
        
        return bug_id, {'status': 'Success', 'details': bug_results}

    except Exception as e:
        logging.error(f"Unhandled exception for bug {bug_id}: {e}")
        return bug_id, {'status': 'Error: Exception', 'details': {'error': str(e), 'traceback': traceback.format_exc()}}
    finally:
        if os.path.exists(unique_checkout_base):
            subprocess.run(['rm', '-rf', unique_checkout_base], check=False)

def execution_tests(adapter: DatasetAdapter, project_path, bug_meta_data, inference_code) -> str:
    target_file_path = os.path.join(project_path, bug_meta_data['file']['file_path'])
    target_file_path_backup = target_file_path + '.backup'
    
    if not os.path.exists(target_file_path):
        logging.error(f"Target file not found: {target_file_path}")
        return "Error: file not found"

    try:
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

        test_flag = adapter.test(project_path)
        return 'Pass' if test_flag == 'Plausible' else 'Fail'

    except Exception as e:
        logging.error(f"Exception during execution_tests for {os.path.basename(project_path)}: {e}", exc_info=True)
        return 'Error: test execution'
    finally:
        if os.path.exists(target_file_path_backup):
            subprocess.run(['mv', target_file_path_backup, target_file_path], check=False)

# --- 유틸리티 함수 (기존과 동일) ---
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

# --- 메인 함수 (멀티프로세싱 조율) ---
def main():
    setup_logging()
    args = get_parser().parse_args()
    adapter = adapter_factory(args.dataset)
    bug_filter = set(args.bug_id_list.split(',')) if args.bug_id_list else None

    # 1) 패치 결과 JSON 로딩 (Results/<dir>/5.PatchesResults_Java.json 우선, 없으면 현재 폴더 폴백)
    evaluate_dir_to_file_dict = {}
    for model_inference_dir in args.model_inference_dirs.split(','):
        model_inference_dir = model_inference_dir.strip()
        if not model_inference_dir:
            continue
        
        evaluate_dir_to_file_dict[model_inference_dir] = {}
        path = os.path.join(MODEL_INFERENCE_BASE_PATH, model_inference_dir, "5.PatchesResults_Java.json")
        alt_path = os.path.join(CURRENT_DIR_PATH, "5.PatchesResults_Java.json")
        if not os.path.exists(path) and os.path.exists(alt_path):
            path = alt_path
        
        if os.path.exists(path):
            logging.info(f"Successfully loaded inference file: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                evaluate_dir_to_file_dict[model_inference_dir][path] = json.load(f)
        else:
            logging.warning(f"Could not find inference file: {path}")

    loaded_any = any(len(v) > 0 for v in evaluate_dir_to_file_dict.values())
    if not loaded_any:
        logging.error("No inference JSON found. "
                      "Put 5.PatchesResults_Java.json under Results/<dir>/ or current folder.")
        raise SystemExit(2)

    # 2) 선행 조건 사전 점검 (오프라인 평가 없음: 없으면 즉시 종료)
    bugs_meta_data_file = f"{PROJECT_DIR_BASE}/defects4j_bugs_meta_data.json"
    if not os.path.exists(bugs_meta_data_file):
        logging.error(f"Missing meta file: {bugs_meta_data_file}")
        raise SystemExit(2)

    if not has_command('defects4j'):
        logging.error("Defects4J CLI not found. Install and ensure it's on PATH.")
        raise SystemExit(2)

    if not has_command('javac'):
        logging.error("JDK (javac) not found. Install JDK and ensure 'javac' is on PATH.")
        raise SystemExit(2)

    # 3) 버그 메타 로딩
    bugs_meta_data: dict = json.load(open(bugs_meta_data_file, 'r', encoding='utf-8'))
    
    # 4) 태스크 준비
    tasks = []
    for bug_id, meta_data in bugs_meta_data.items():
        if bug_filter and bug_id not in bug_filter:
            continue
        if adapter.should_skip_bug(bug_id):
            logging.info(f"Adapter policy: Skipping bug {bug_id}")
            continue
        tasks.append((bug_id, meta_data, adapter, evaluate_dir_to_file_dict))

    num_processes = max(1, multiprocessing.cpu_count() - 1)
    logging.info(f"Starting parallel evaluation with {num_processes} processes for {len(tasks)} bugs.")
    
    all_results = {}
    failed_bugs = {}
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        for bug_id, result_data in pool.map(evaluate_bug_worker, tasks):
            if result_data['status'] == 'Success':
                all_results[bug_id] = result_data['details']
            else:
                failed_bugs[bug_id] = result_data

    logging.info("All parallel evaluations finished. Aggregating and writing results...")

    # 5) 결과 저장
    for model_inference_dir, path_file_dict in evaluate_dir_to_file_dict.items():
        _evaluate_path = os.path.join(CURRENT_DIR_PATH, adapter.dataset_name, sanitize_name(model_inference_dir))
        os.makedirs(_evaluate_path, exist_ok=True)
        
        for path in path_file_dict.keys():
            name_suf = os.path.splitext(os.path.basename(path))[0]
            evaluate_path = os.path.join(_evaluate_path, f"unittest_result_{name_suf}.json")
            pass_k_result_path = os.path.join(_evaluate_path, f"pass_k_result_{name_suf}.txt")

            final_json_output = {}
            pass_k_logs = []

            for bug_id, bug_results in all_results.items():
                if path in bug_results:
                    final_json_output[bug_id] = bug_results[path]
                    
                    bug_meta_data = bugs_meta_data[bug_id]
                    bug_id_key = 'bugsinpy_id' if adapter.dataset_name == 'bugsinpy' else 'defects4j_id'
                    project_name = adapter.map_project_name(bug_meta_data['project_name'])
                    
                    log_msg = f"bug id {bug_id} ({project_name}-{bug_meta_data[bug_id_key]}) unittest result: {bug_results[path]['nucleus_sampling_flags']}"
                    pass_k_logs.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {log_msg}\n")
            
            with open(evaluate_path, 'w', encoding='utf-8') as f:
                json.dump(final_json_output, f, indent=2)

            with open(pass_k_result_path, 'w', encoding='utf-8') as f:
                f.write("=========================================\n")
                f.writelines(pass_k_logs)

    logging.error(f"Bugs that failed during processing: {json.dumps(failed_bugs, indent=2)}")
    logging.info("ALL unittests finished!")

if __name__ == '__main__':
    # 멀티프로세싱 시작점 설정 (macOS/Windows 호환성)
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass  # 이미 설정된 경우 무시
    main()
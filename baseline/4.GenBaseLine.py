import json
import sys
import copy
import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# --- 기본값 (CLI로 덮어쓰기 가능) ---
BASE_DIR = Path("./Results")
DEFAULT_INPUT_JSON_PATH = BASE_DIR / "1.DataSeterAgentResult.json"
DEFAULT_OUTPUT_JSON_PATH = BASE_DIR / "4.BaseLine.json"

DEFAULT_MAX_CODE_LINES = 200     # CODE_SNIPPET 최대 줄수
DEFAULT_MAX_FULL_CODE_LINES = 1000  # 함수 전체를 넣을 때 상한


# --- 유틸리티 ---

def read_json_file(file_path: Path) -> Optional[Union[Dict[str, Any], List[Any]]]:
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"❌ 오류: 파일을 찾을 수 없습니다 '{file_path}'")
    except json.JSONDecodeError:
        print(f"❌ 오류: JSON 파싱에 실패했습니다 '{file_path}'")
    return None


def write_json_file(data: Union[Dict[str, Any], List[Any]], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def trim_text_by_lines(text: Optional[str], max_lines: int) -> str:
    if not text:
        return ""
    # 통일된 개행 처리
    lines = text.replace("\r\n", "\n").splitlines()
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + "\n... (truncated)"
    return "\n".join(lines)


def safe_get(data: Any, path: List[Any], default: Any = None) -> Any:
    current = data
    for key in path:
        try:
            current = current[key]
        except (KeyError, IndexError, TypeError):
            return default
    return current


def _as_str(x: Any, fallback: str = "") -> str:
    return x if isinstance(x, str) else fallback


# --- 데이터 추출 (베이스라인: 최소 정보만) ---

def get_buggy_function_code(bug_data: Dict[str, Any]) -> str:
    """
    베이스라인에서는 '버그 코드'만 제공.
    우선순위:
      1) bug_data['function']['function_before']
      2) bug_data['bm25']['top'][0]['function']['function_before'] (백업)
    """
    code_sources = [
        safe_get(bug_data, ["function", "function_before"]),
        safe_get(bug_data, ["bm25", "top", 0, "function", "function_before"]),
    ]
    for code in code_sources:
        if isinstance(code, str) and code.strip():
            return code
    return ""


def get_code_snippet(bug_data: Dict[str, Any], max_code_lines: int, max_full_code_lines: int) -> str:
    """
    가능한 경우 함수 전체를 사용하되, 너무 길면 잘라서 제공.
    (베이스라인이므로 추가 표식/주석 등은 넣지 않음)
    """
    full_buggy_code = get_buggy_function_code(bug_data)
    if not full_buggy_code:
        # 혹시 스니펫 필드가 따로 있다면 여기에 추가로 탐색 가능
        full_buggy_code = _as_str(safe_get(bug_data, ["snippet"]), "")
    if not full_buggy_code:
        return ""

    # 우선 전체 상한으로 한 번 자르고, 그래도 길면 더 작은 상한으로 재자름
    clipped = trim_text_by_lines(full_buggy_code, max_full_code_lines)
    if clipped.endswith("... (truncated)"):
        # 너무 크면 더 작은 스니펫 상한으로 축소
        clipped = trim_text_by_lines(full_buggy_code, max_code_lines)
    return clipped


# --- 프롬프트 생성 (베이스라인 제로샷 / 패치만 출력) ---

BASELINE_PROMPT_TEMPLATE = """
"You are an expert Python programming assistant. Your task is to analyze the provided buggy Python function and generate a corrected version.

 CRITICAL INSTRUCTIONS:
- Output only the patch. Do not include explanations, rationale, or commit messages.
- Do not change function or class signatures unless absolutely necessary.
- Keep the patch minimal and consistent with the existing style.
- If multiple fixes are possible, choose the simplest and least invasive.
- If uncertain, output the most plausible patch.

###Bug INFO 
BUG_ID: {bug_id}
PROJECT: {project_name}
FILE_PATH: {file_path}
FUNCTION_NAME: {function_name}
LINE_WITH_BUG: {buggy_line}
CODE_SNIPPET:
{code_snippet}
###END BUG INFO

Now, generate your response following the critical instructions above.
        ##correct
"""


def create_prompt(
    bug_id: str,
    bug_data: Dict[str, Any],
    max_code_lines: int,
    max_full_code_lines: int,
) -> str:
    project_name = str(bug_data.get("project_name", "N/A"))
    file_path = str(bug_data.get("file_path", "N/A"))
    function_name = str(safe_get(bug_data, ["function", "function_name"], "N/A"))
    buggy_line_content = (_as_str(bug_data.get("buggy_line_content"), "") or "").strip()

    code_snippet = get_code_snippet(
        bug_data=bug_data,
        max_code_lines=max_code_lines,
        max_full_code_lines=max_full_code_lines,
    )
    # CODE_SNIPPET은 코드블록 없이 그대로 넣는다 (모델이 그대로 읽고 diff 생성하게)
    # 필요 시 ```python 감싸도 되지만, 여기서는 최대한 '정보 최소화' 원칙 유지.

    prompt = BASELINE_PROMPT_TEMPLATE.format(
        bug_id=bug_id,
        project_name=project_name,
        file_path=file_path,
        function_name=function_name,
        buggy_line=buggy_line_content,
        code_snippet=code_snippet,
    )
    return prompt


def normalize_top_level(json_obj: Union[Dict[str, Any], List[Any]]) -> Dict[str, Any]:
    """
    입력 JSON이 dict 또는 list 모두를 허용.
    - dict: 그대로 사용
    - list: 인덱스를 키로 하는 dict로 변환
    """
    if isinstance(json_obj, dict):
        return json_obj
    if isinstance(json_obj, list):
        return {str(i): v for i, v in enumerate(json_obj)}
    raise TypeError("Top-level JSON must be an object or an array.")


def process_file(
    input_path: Path,
    output_path: Path,
    max_code_lines: int,
    max_full_code_lines: int,
) -> int:
    source = read_json_file(input_path)
    if source is None:
        return 1

    try:
        processed: Dict[str, Any] = copy.deepcopy(normalize_top_level(source))
    except Exception as e:
        print(f"❌ 입력 JSON 최상위 스키마 변환 실패: {type(e).__name__}: {e}")
        return 1

    errors: Dict[str, str] = {}

    for bug_id, bug_data in processed.items():
        try:
            if not isinstance(bug_data, dict):
                raise TypeError(f"bug '{bug_id}' is not an object")
            processed[bug_id]["prompt"] = create_prompt(
                bug_id,
                bug_data,
                max_code_lines=max_code_lines,
                max_full_code_lines=max_full_code_lines,
            )
        except Exception as e:
            errors[bug_id] = f"{type(e).__name__}: {e}"
            processed[bug_id]["prompt_error"] = errors[bug_id]

    write_json_file(processed, output_path)

    if errors:
        print(f"⚠️ 일부 항목 생성 실패({len(errors)}건). 'prompt_error' 필드 참조.")
    print(f"✅ 프롬프트가 추가된 JSON 파일 저장 완료: {output_path}")
    return 0 if not errors else 1  # 오류가 하나라도 있으면 실패(1)를 반환


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate baseline zero-shot prompts for bug-fix tasks (code-only patch).")
    p.add_argument("-i", "--input", default=str(DEFAULT_INPUT_JSON_PATH), help="입력 JSON 경로")
    p.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT_JSON_PATH), help="출력 JSON 경로")
    p.add_argument("--max-code-lines", type=int, default=DEFAULT_MAX_CODE_LINES, help="코드 스니펫 최대 줄수")
    p.add_argument("--max-full-code-lines", type=int, default=DEFAULT_MAX_FULL_CODE_LINES, help="전체 함수 코드 최대 줄수")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rc = process_file(
        input_path=Path(args.input),
        output_path=Path(args.output),
        max_code_lines=args.max_code_lines,
        max_full_code_lines=args.max_full_code_lines,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
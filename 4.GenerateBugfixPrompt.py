import json
import sys
import copy
import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# --- 기본값 (CLI로 덮어쓰기 가능) ---
BASE_DIR = Path("./Results")
DEFAULT_INPUT_JSON_PATH = BASE_DIR / "3.BM25Result.json"
DEFAULT_OUTPUT_JSON_PATH = BASE_DIR / "4.GeneratePromResult.json"

DEFAULT_MAX_SIMILAR_EXAMPLES = 1
DEFAULT_MAX_CODE_LINES = 200
DEFAULT_MAX_LIST_LINES = 30
DEFAULT_MAX_FULL_CODE_LINES = 1000


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


# --- 데이터 추출/요약 ---

def get_buggy_function_code(bug_data: Dict[str, Any]) -> str:
    code_sources = [
        safe_get(bug_data, ["function", "function_before"]),
        safe_get(bug_data, ["bm25", "top", 0, "function", "function_before"]),
    ]
    for code in code_sources:
        if isinstance(code, str) and code.strip():
            return code
    return ""


def get_similar_fix_examples(bug_data: Dict[str, Any], count: int) -> List[Dict[str, str]]:
    if count <= 0:
        return []
    examples: List[Dict[str, str]] = []
    similar_items = safe_get(bug_data, ["bm25", "top"], []) or []
    for item in similar_items:
        function_data = item.get("function", {}) or {}
        before_code = function_data.get("function_before")
        after_code = function_data.get("function_after")
        if all(isinstance(code, str) and code.strip() for code in [before_code, after_code]):
            examples.append({
                "project_name": str(item.get("project_name", "unknown")),
                "file_path": str(item.get("file_path", "")),
                "function_name": str(function_data.get("function_name", "")),
                "before_code": before_code,
                "after_code": after_code,
            })
        if len(examples) >= count:
            break
    return examples


def summarize_commit_info(bug_data: Dict[str, Any], max_code_lines: int) -> str:
    commit_message = (_as_str(safe_get(bug_data, ["commit", "commit_message"]), "") or "").strip()
    commit_diff = trim_text_by_lines(safe_get(bug_data, ["commit", "commit_file_diff"], ""), max_code_lines)
    parts: List[str] = []
    if commit_message:
        parts.append(f"- Message: {commit_message}")
    if commit_diff:
        parts.append(f"- Diff:\n```diff\n{commit_diff}\n```")
    return "\n".join(parts)


def summarize_suspicious_nodes(bug_data: Dict[str, Any], max_list_lines: int) -> str:
    nodes = safe_get(bug_data, ["suspicious_nodes_topk"], []) or []
    if not nodes:
        return ""
    summary_parts = []
    for i, node in enumerate(nodes[:5], 1):
        line = node.get("line", "N/A")
        node_type = node.get("type", "N/A")
        code_snippet = trim_text_by_lines(_as_str(node.get("code"), ""), 5).strip()
        reason = _as_str(node.get("reason"), "")
        # 인라인 백틱 충돌 최소화를 위해 코드블록 대신 짧은 한 줄 요약 + 원문 라인 제한
        summary_parts.append(
            f"{i}. **Line {line}** (type: `{node_type}`)\n"
            f"   - Code:\n```python\n{code_snippet}\n```\n"
            f"   - Reason: {reason}"
        )
    return trim_text_by_lines("\n".join(summary_parts), max_list_lines)


def _mark_buggy_line(full_code: str, buggy_line_content: str) -> str:
    """
    보다 안전하게 버그 라인을 표시:
    1) 완전 일치(공백 제외) 우선
    2) 실패 시 부분 포함 첫 라인에 표시
    """
    if not (full_code and buggy_line_content):
        return full_code

    lines = full_code.split("\n")
    target = buggy_line_content.strip()

    # 1) 완전 일치(좌우 공백 무시)
    for i, line in enumerate(lines):
        if line.strip() == target:
            lines[i] = f"{line}  # <--- BUGGY LINE"
            return "\n".join(lines)

    # 2) 부분 포함 (가장 먼저 등장하는 곳)
    for i, line in enumerate(lines):
        if target and target in line:
            lines[i] = f"{line}  # <--- BUGGY LINE"
            return "\n".join(lines)

    return full_code


def create_context_string(bug_data: Dict[str, Any], max_similar: int, max_code_lines: int, max_list_lines: int) -> str:
    pieces: List[str] = []

    # Similar Fix Examples (복수 개)
    examples = get_similar_fix_examples(bug_data, max_similar)
    if examples:
        blocks = []
        for idx, ex in enumerate(examples, 1):
            blocks.append(
                f"#### Example {idx}\n"
                f"- Project: `{ex['project_name']}`\n"
                f"- File: `{ex['file_path']}`\n"
                f"- Function: `{ex['function_name']}`\n\n"
                f"**Buggy Version:**\n"
                f"```python\n{trim_text_by_lines(ex['before_code'], max_code_lines)}\n```\n\n"
                f"**Fixed Version:**\n"
                f"```python\n{trim_text_by_lines(ex['after_code'], max_code_lines)}\n```"
            )
        pieces.append("### Context 1: Similar Fix Examples\n" + "\n\n".join(blocks))

    # Commit info
    commit_summary = summarize_commit_info(bug_data, max_code_lines)
    if commit_summary:
        pieces.append("### Context 2: Related Commit Info\n" + commit_summary)

    # Suspicious nodes
    susp = summarize_suspicious_nodes(bug_data, max_list_lines)
    if susp:
        pieces.append("### Context 3: Static Analysis Hints\n" + susp)

    if not pieces:
        return "No additional context is available."
    return "\n\n---\n\n".join(pieces)


# --- 프롬프트 생성 ---

def create_prompt(
    bug_id: str,
    bug_data: Dict[str, Any],
    max_similar: int,
    max_code_lines: int,
    max_list_lines: int,
    max_full_code_lines: int,
) -> str:
    project_name = str(bug_data.get("project_name", "N/A"))
    file_path = str(bug_data.get("file_path", "N/A"))
    function_name = str(safe_get(bug_data, ["function", "function_name"], "N/A"))
    buggy_line_content = (_as_str(bug_data.get("buggy_line_content"), "") or "").strip()
    description = str(bug_data.get("description", "No description provided."))
    full_buggy_code = get_buggy_function_code(bug_data)
    context_string = create_context_string(bug_data, max_similar, max_code_lines, max_list_lines)

    if full_buggy_code and buggy_line_content:
        full_buggy_code = _mark_buggy_line(full_buggy_code, buggy_line_content)

    return (
        "You are an expert Python programming assistant. Your task is to analyze the provided buggy Python function and generate a corrected version.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1.  First, in an <analysis> XML tag, briefly explain the root cause of the bug and your planned fix.\n"
        "2.  After the analysis tag, provide ONLY the complete, corrected function or class. It must be a drop-in replacement.\n"
        "3.  Do not include import statements or any introductory text before the code itself.\n"
        "4.  Do not change the original function name or its signature.\n"
        "5.  Pay extreme attention to indentation. The final code must be syntactically correct.\n"
        "6.  The fix should only target the bug with minimal changes. Preserve all existing correct logic.\n"
        "7.  The code must be fully implemented without placeholders like pass or ....\n"
        "8.  The entire output, including any variable names or strings, must be in English and compatible with Python 3.6.\n\n"
        "### Bug INFO\n\n"
        f"**Project:** `{project_name}`\n"
        f"**File:** `{file_path}`\n"
        f"**Function to Fix:** `{function_name}`\n"
        f"**Identified Buggy Line:** `{buggy_line_content}`\n\n"
        "**Bug Description:**\n"
        f"{trim_text_by_lines(description, 80)}\n\n"
        "**Full Buggy Code** (The problematic line is marked with `# <--- BUGGY LINE`):\n"
        "```python\n"
        f"{trim_text_by_lines(full_buggy_code, max_full_code_lines)}\n"
        "```\n\n"
        "### Reference Context\n"
        "Use the following context to understand the bug and formulate the fix. Pay special attention to the **Static Analysis Hints** as they often point to the direct cause, and use the **Similar Fix Examples** to find common patching patterns.\n\n"
        f"{context_string}\n\n"
        "Now, generate your response following the critical instructions above.\n"
        "##correct\n"
    )


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
    max_similar: int,
    max_code_lines: int,
    max_list_lines: int,
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
                max_similar=max_similar,
                max_code_lines=max_code_lines,
                max_list_lines=max_list_lines,
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
    p = argparse.ArgumentParser(description="Generate LLM prompts for bug-fix tasks.")
    p.add_argument("-i", "--input", default=str(DEFAULT_INPUT_JSON_PATH), help="입력 JSON 경로")
    p.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT_JSON_PATH), help="출력 JSON 경로")
    p.add_argument("--max-similar", type=int, default=DEFAULT_MAX_SIMILAR_EXAMPLES, help="유사 사례 최대 개수")
    p.add_argument("--max-code-lines", type=int, default=DEFAULT_MAX_CODE_LINES, help="코드 스니펫 최대 줄수")
    p.add_argument("--max-list-lines", type=int, default=DEFAULT_MAX_LIST_LINES, help="리스트 요약 최대 줄수")
    p.add_argument("--max-full-code-lines", type=int, default=DEFAULT_MAX_FULL_CODE_LINES, help="전체 함수 코드 최대 줄수")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rc = process_file(
        input_path=Path(args.input),
        output_path=Path(args.output),
        max_similar=args.max_similar,
        max_code_lines=args.max_code_lines,
        max_list_lines=args.max_list_lines,
        max_full_code_lines=args.max_full_code_lines,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
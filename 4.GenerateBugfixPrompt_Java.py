#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import sys
import copy
import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# --- 기본값 (CLI로 덮어쓰기 가능) ---
BASE_DIR = Path("./Results")
DEFAULT_INPUT_JSON_PATH = BASE_DIR / "3.BM25Result_Java.json"
DEFAULT_OUTPUT_JSON_PATH = BASE_DIR / "4.GeneratePromResult_Java.json"

DEFAULT_MAX_SIMILAR_EXAMPLES = 1
DEFAULT_MAX_CODE_LINES = 200
DEFAULT_MAX_LIST_LINES = 30
DEFAULT_MAX_FULL_CODE_LINES = 1000

# 멀티-변이 기본
DEFAULT_OUT_PREFIX = "4.GeneratePromResult__"
VARIANTS = {
    "full":            {"use_ast": True,  "use_retrieval": True},
    "wo_ast":          {"use_ast": False, "use_retrieval": True},
    "wo_retrieval":    {"use_ast": True,  "use_retrieval": False},
    "ast_only":        {"use_ast": True,  "use_retrieval": False},  # alias
    "retrieval_only":  {"use_ast": False, "use_retrieval": True},   # alias
    "none":            {"use_ast": False, "use_retrieval": False},
}

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
    for i, node in enumerate(nodes[:5], 1):  # 기존 스타일 유지: 상위 5개
        line = node.get("line", "N/A")
        node_type = node.get("type", "N/A")
        code_snippet = trim_text_by_lines(_as_str(node.get("code"), ""), 5).strip()
        reason = _as_str(node.get("reason"), "")
        summary_parts.append(
            f"{i}. **Line {line}** (type: `{node_type}`)\n"
            f"   - Code:\n```Java\n{code_snippet}\n```\n"
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

def create_context_string(
    bug_data: Dict[str, Any],
    max_similar: int,
    max_code_lines: int,
    max_list_lines: int,
    use_ast: bool,
    use_retrieval: bool,
) -> str:
    pieces: List[str] = []

    # Similar Fix Examples (복수 개) - 토글
    if use_retrieval:
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
                    f"```Java\n{trim_text_by_lines(ex['before_code'], max_code_lines)}\n```\n\n"
                    f"**Fixed Version:**\n"
                    f"```Java\n{trim_text_by_lines(ex['after_code'], max_code_lines)}\n```"
                )
            pieces.append("### Context 1: Similar Fix Examples\n" + "\n\n".join(blocks))

    # Commit info (항상 포함: 토글 요구사항에 없음)
    commit_summary = summarize_commit_info(bug_data, max_code_lines)
    if commit_summary:
        pieces.append("### Context 2: Related Commit Info\n" + commit_summary)

    # Suspicious nodes (AST 힌트) - 토글
    if use_ast:
        susp = summarize_suspicious_nodes(bug_data, max_list_lines)
        if susp:
            pieces.append("### Context 3: Static Analysis Hints\n" + susp)

    if not pieces:
        return "No additional context is available."
    return "\n\n---\n\n".join(pieces)

# --- 프롬프트 생성(강화 버전) ---

def create_prompt(
    bug_id: str,
    bug_data: Dict[str, Any],
    max_similar: int,
    max_code_lines: int,
    max_list_lines: int,
    max_full_code_lines: int,
    use_ast: bool,
    use_retrieval: bool,
) -> str:
    project_name = str(bug_data.get("project_name", "N/A"))
    file_path = str(bug_data.get("file_path", "N/A"))
    function_name = str(safe_get(bug_data, ["function", "function_name"], "N/A"))
    buggy_line_content = (_as_str(bug_data.get("buggy_line_content"), "") or "").strip()
    description = str(bug_data.get("description", "No description provided."))
    full_buggy_code = get_buggy_function_code(bug_data)
    context_string = create_context_string(
        bug_data, max_similar, max_code_lines, max_list_lines,
        use_ast=use_ast, use_retrieval=use_retrieval
    )

    if full_buggy_code and buggy_line_content:
        full_buggy_code = _mark_buggy_line(full_buggy_code, buggy_line_content)

    # 단호하고 간결한 지시 + 체크리스트 + 탐색 우선순위
    return (
        "You are a senior Java bug-fix specialist. Read the task and produce a minimal, correct patch.\n\n"
        "OUTPUT PROTOCOL (strict):\n"
        "1) First, output an <analysis> XML block (short: <= 12 lines).\n"
        "   - State the SINGLE root cause.\n"
        "   - Point to the exact line(s) causing the failure.\n"
        "   - Describe the ONE minimal change you will apply and why it is safe.\n"
        "   - Run this mental CHECKLIST before finalizing:\n"
        "     [ ] Off-by-one / range / indexing\n"
        "     [ ] None / empty ([], {}, \"\") handling\n"
        "     [ ] Mutable default args / aliasing (copy vs reference)\n"
        "     [ ] Type coercion (str/int/float), truthiness pitfalls\n"
        "     [ ] Sorting / order assumptions (stable sort, key)\n"
        "     [ ] Integer division vs float division\n"
        "     [ ] Early return vs fall-through logic\n"
        "     [ ] Exception type/messages preserved or narrowed safely\n"
        "     [ ] Boundary conditions on loops/slices\n"
        "     [ ] Do not change I/O behavior or global state\n"
        "2) After </analysis>, output ONLY the complete corrected function or class in a Java code block.\n\n"
        "STRICT CONSTRAINTS:\n"
        "- Do not add imports. Do not print. Do not log. No placeholders like 'pass'.\n"
        "- Keep the original name and signature EXACTLY.\n"
        "- Apply the smallest possible edit (prefer changing <= 3 lines unless absolutely necessary).\n"
        "- Preserve behavior that is unrelated to the bug.\n"
        "- Java old version compatible. All identifiers and strings must be in English.\n"
        "- If the bug is in conditionals/boundaries, fix the condition not the data.\n"
        "- If the bug involves mutation vs copy, use an explicit shallow copy only when required.\n"
        "- If external APIs are used, do NOT alter their contract or error types.\n\n"
        "FOCUS ORDER WHEN READING CONTEXT:\n"
        "1) Static Analysis Hints (suspicious nodes) — inspect these lines first.\n"
        "2) The line marked with '# <--- BUGGY LINE'.\n"
        "3) Surrounding lines for dataflow and invariants.\n"
        "4) Similar Fix Examples — only to recognize patterns, not to rewrite wholesale.\n\n"
        "### Bug INFO\n\n"
        f"**Project:** `{project_name}`\n"
        f"**File:** `{file_path}`\n"
        f"**Function to Fix:** `{function_name}`\n"
        f"**Identified Buggy Line:** `{buggy_line_content}`\n\n"
        "**Bug Description (brief):**\n"
        f"{trim_text_by_lines(description, 60)}\n\n"
        "**Full Buggy Code** (the problematic line is marked with `# <--- BUGGY LINE`):\n"
        "```Java\n"
        f"{trim_text_by_lines(full_buggy_code, max_full_code_lines)}\n"
        "```\n\n"
        "### Reference Context\n"
        "Use the context below only to confirm the hypothesis and to borrow minimal patch patterns.\n"
        "Avoid large refactors; prefer a one-line or small-guard fix when valid.\n\n"
        f"{context_string}\n\n"
        "Now produce exactly TWO blocks as specified in the OUTPUT PROTOCOL.\n"
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

def process_content(
    source: Union[Dict[str, Any], List[Any]],
    use_ast: bool,
    use_retrieval: bool,
    max_similar: int,
    max_code_lines: int,
    max_list_lines: int,
    max_full_code_lines: int,
) -> Dict[str, Any]:
    try:
        processed: Dict[str, Any] = copy.deepcopy(normalize_top_level(source))
    except Exception as e:
        raise RuntimeError(f"입력 JSON 최상위 스키마 변환 실패: {type(e).__name__}: {e}")

    errors: Dict[str, str] = {}

    for bug_id, bug_data in processed.items():
        try:
            if not isinstance(bug_data, dict):
                raise TypeError(f"bug '{bug_id}' is not an object")
            prompt = create_prompt(
                bug_id,
                bug_data,
                max_similar=max_similar,
                max_code_lines=max_code_lines,
                max_list_lines=max_list_lines,
                max_full_code_lines=max_full_code_lines,
                use_ast=use_ast,
                use_retrieval=use_retrieval,
            )
            processed[bug_id]["prompt"] = prompt
            processed[bug_id]["ablation"] = {
                "use_ast": bool(use_ast),
                "use_retrieval": bool(use_retrieval),
            }
        except Exception as e:
            errors[bug_id] = f"{type(e).__name__}: {e}"
            processed[bug_id]["prompt_error"] = errors[bug_id]

    if errors:
        print(f"⚠️ 일부 항목 생성 실패({len(errors)}건). 'prompt_error' 필드 참조.")
    return processed

def process_file_single(
    input_path: Path,
    output_path: Path,
    use_ast: bool,
    use_retrieval: bool,
    max_similar: int,
    max_code_lines: int,
    max_list_lines: int,
    max_full_code_lines: int,
) -> int:
    source = read_json_file(input_path)
    if source is None:
        return 1
    try:
        processed = process_content(
            source, use_ast, use_retrieval,
            max_similar, max_code_lines, max_list_lines, max_full_code_lines
        )
    except Exception as e:
        print(f"❌ 처리 실패: {e}")
        return 1
    write_json_file(processed, output_path)
    print(f"✅ 프롬프트가 추가된 JSON 파일 저장 완료: {output_path}")
    return 0

# --- CLI ---

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate LLM prompts for bug-fix tasks (AST/Retrieval 토글 및 멀티-변이 지원).")
    p.add_argument("-i", "--input", default=str(DEFAULT_INPUT_JSON_PATH), help="입력 JSON 경로")
    p.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT_JSON_PATH), help="(싱글 모드) 출력 JSON 경로")
    p.add_argument("--max-similar", type=int, default=DEFAULT_MAX_SIMILAR_EXAMPLES, help="유사 사례 최대 개수")
    p.add_argument("--max-code-lines", type=int, default=DEFAULT_MAX_CODE_LINES, help="코드 스니펫 최대 줄수")
    p.add_argument("--max-list-lines", type=int, default=DEFAULT_MAX_LIST_LINES, help="리스트 요약 최대 줄수")
    p.add_argument("--max-full-code-lines", type=int, default=DEFAULT_MAX_FULL_CODE_LINES, help="전체 함수 코드 최대 줄수")

    # 토글
    p.add_argument("--use-ast", type=int, choices=[0,1], default=1, help="AST 신호(의심 노드) 포함 여부")
    p.add_argument("--use-retrieval", type=int, choices=[0,1], default=1, help="BM25 유사 예시 포함 여부")

    # 멀티-변이
    p.add_argument("--variants", type=str, default="", help="콤마 구분 변이들: full,wo_ast,wo_retrieval,ast_only,retrieval_only,none")
    p.add_argument("--out-dir", type=str, default=str(BASE_DIR), help="(멀티 모드) 출력 폴더")
    p.add_argument("--out-prefix", type=str, default=DEFAULT_OUT_PREFIX, help="(멀티 모드) 파일 접두어")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    in_path = Path(args.input)

    if not args.variants.strip():
        # 싱글 모드
        rc = process_file_single(
            input_path=in_path,
            output_path=Path(args.output),
            use_ast=bool(args.use_ast),
            use_retrieval=bool(args.use_retrieval),
            max_similar=args.max_similar,
            max_code_lines=args.max_code_lines,
            max_list_lines=args.max_list_lines,
            max_full_code_lines=args.max_full_code_lines,
        )
        sys.exit(rc)
    else:
        # 멀티-변이 모드
        source = read_json_file(in_path)
        if source is None:
            sys.exit(1)

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        names = [v.strip() for v in args.variants.split(",") if v.strip()]
        rc_any_fail = 0
        for name in names:
            if name not in VARIANTS:
                print(f"⚠️ 알 수 없는 변이 '{name}' — 건너뜀")
                continue
            cfg = VARIANTS[name]
            try:
                processed = process_content(
                    source=source,
                    use_ast=cfg["use_ast"],
                    use_retrieval=cfg["use_retrieval"],
                    max_similar=args.max_similar,
                    max_code_lines=args.max_code_lines,
                    max_list_lines=args.max_list_lines,
                    max_full_code_lines=args.max_full_code_lines,
                )
                out_path = out_dir / f"{args.out_prefix}{name}.json"
                write_json_file(processed, out_path)
                print(f"✅ [{name}] 저장 완료: {out_path}")
            except Exception as e:
                print(f"❌ [{name}] 처리 실패: {e}")
                rc_any_fail = 1

        sys.exit(rc_any_fail)

if __name__ == "__main__":
    main()

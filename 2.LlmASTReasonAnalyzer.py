import argparse
import ast
import json
import os
import re
import random
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

# LLM 및 딥러닝 라이브러리
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

# ‼️ 수정: Python 3.8 호환성을 위해 astunparse 라이브러리 import
import astunparse


# -------------------- AST 유틸리티 --------------------

def ast_to_dict(node: ast.AST) -> Any:
    """AST 노드를 파이썬 dict로 재귀 변환."""
    if isinstance(node, ast.AST):
        result = {'_type': type(node).__name__}
        for field in getattr(node, "_fields", ()):
            value = getattr(node, field, None)
            if isinstance(value, list):
                result[field] = [ast_to_dict(item) for item in value]
            else:
                result[field] = ast_to_dict(value)
        return result
    elif isinstance(node, list):
        return [ast_to_dict(item) for item in node]
    elif isinstance(node, bytes):
        try:
            return node.decode("utf-8", errors="replace")
        except Exception:
            return str(node)
    else:
        return node


def extract_suspicious_nodes_with_ast(function_code: str) -> List[Dict[str, Any]]:
    """의심 노드+AST dict 함께 추출"""

    class NodeVisitor(ast.NodeVisitor):
        def __init__(self):
            self.nodes = []

        def add(self, node, nodetype):
            self.nodes.append({
                "type": nodetype,
                "line": getattr(node, "lineno", None),
                # ‼️ 수정: ast.unparse 대신 astunparse.unparse 사용
                "code": astunparse.unparse(node).strip(),
                "ast_subtree_json": ast_to_dict(node),
            })

        def visit_For(self, node): self.add(node, "For"); self.generic_visit(node)

        def visit_If(self, node): self.add(node, "If"); self.generic_visit(node)

        def visit_While(self, node): self.add(node, "While"); self.generic_visit(node)

        def visit_Call(self, node): self.add(node, "Call"); self.generic_visit(node)

        def visit_Assign(self, node): self.add(node, "Assign"); self.generic_visit(node)

        def visit_Return(self, node): self.add(node, "Return"); self.generic_visit(node)

        def visit_With(self, node): self.add(node, "With"); self.generic_visit(node)

        def visit_Try(self, node): self.add(node, "Try"); self.generic_visit(node)

        def visit_ExceptHandler(self, node): self.add(node, "ExceptHandler"); self.generic_visit(node)

    try:
        tree = ast.parse(function_code)
    except Exception as e:
        return [{"error": str(e)}]
    visitor = NodeVisitor()
    visitor.visit(tree)
    return visitor.nodes


def get_code_context(source: str, lineno: Optional[int], window: int = 3) -> str:
    lines = source.splitlines()
    if lineno is None or lineno < 1 or lineno > len(lines):
        return ""
    start = max(0, lineno - 1 - window)
    end = min(len(lines), lineno + window)
    context_lines = []
    for i in range(start, end):
        prefix = "-> " if i == lineno - 1 else "   "
        context_lines.append(f"{i + 1:4d}:{prefix}{lines[i]}")
    return "\n".join(context_lines)


# -------------------- LLM 헬퍼 --------------------

_JSON_ARRAY_RE = re.compile(r'\[\s*\{.*?\}\s*\]', re.DOTALL)


def extract_last_json_array(text: str):
    arrays = list(_JSON_ARRAY_RE.finditer(text))
    if arrays:
        try:
            return json.loads(arrays[-1].group())
        except Exception:
            return []
    return []


def build_messages(code_context: str, node: Dict[str, Any]) -> List[Dict[str, str]]:
    user_prompt = f"""
For the suspicious node below, briefly explain WHY it might be buggy based on the surrounding code context.
Return ONLY a JSON array with ONE object. Keys: "line", "reason".

Example:
[{{"line": 15, "reason": "May cause IndexError if the list is empty."}}]

Python code snippet:
{code_context}

Candidate suspicious node:
{json.dumps([node], indent=2, ensure_ascii=False)}
""".strip()
    return [
        {"role": "system", "content": "You are an expert Python static analyzer."},
        {"role": "user", "content": user_prompt},
    ]


@torch.no_grad()
def query_llm(tokenizer, model, messages, sampling_params: Dict[str, Any]) -> Dict[str, Any]:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    outputs = model.generate(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=256,
        pad_token_id=tokenizer.eos_token_id,
        **sampling_params,
    )
    generated_ids = outputs[0][len(model_inputs.input_ids[0]):]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True)

    parsed = extract_last_json_array(result)
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and "line" in parsed[0] and "reason" in \
            parsed[0]:
        return parsed[0]
    return {"line": None, "reason": f"No valid JSON answer from LLM. Raw output: {result}"}


# -------------------- 랭킹 / 제어 --------------------

def distance_to_buggy_line(node_line: Optional[int], buggy_line: Optional[int]) -> Optional[int]:
    if node_line is None or buggy_line is None:
        return None
    return abs(node_line - buggy_line)


def rank_nodes(nodes: List[Dict[str, Any]], buggy_line: Optional[int]) -> List[Dict[str, Any]]:
    type_weight = {
        "Call": 0, "Assign": 1, "If": 1, "Return": 2, "For": 2, "While": 2,
        "Try": 1, "ExceptHandler": 1, "With": 2
    }
    scored = []
    for n in nodes:
        line = n.get("line")
        dist = distance_to_buggy_line(line, buggy_line)
        weight = type_weight.get(n.get("type"), 3)
        score = (dist if dist is not None else 9999) * 10 + weight
        n2 = dict(n)
        n2["_score"] = score
        scored.append(n2)
    scored.sort(key=lambda x: x["_score"])
    return scored


# -------------------- 메인 분석 로직 --------------------

def _to_int_or_none(v) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def analyze_one_bug(
        bug_id: str,
        item: Dict[str, Any],
        model=None,
        tokenizer=None,
        sampling_params: Optional[Dict[str, Any]] = None,
        topk: int = 10
) -> Dict[str, Any]:
    """단일 버그 분석 (원본 데이터 + 생성된 값 결합)"""
    function_info = item.get("function", {}) or {}
    function_before = function_info.get("function_before", "") or ""
    buggy_line_location = _to_int_or_none(item.get("buggy_line_location"))

    if not function_before:
        return {"error": f"bug_id={bug_id} has empty function_before", **item}

    nodes = extract_suspicious_nodes_with_ast(function_before)
    if nodes and "error" in nodes[0]:
        return {"error": f"AST parsing failed: {nodes[0]['error']}", **item}

    nodes_ranked = rank_nodes(nodes, buggy_line_location)
    nodes_topk = nodes_ranked[:topk]

    suspicious_results = []
    for node in nodes_topk:
        line = node.get("line")
        code_context = get_code_context(function_before, line, window=3)
        reason = "LLM not executed"
        if model and tokenizer and sampling_params:
            msg = build_messages(code_context, node)
            out = query_llm(tokenizer, model, msg, sampling_params)
            reason = out.get("reason", reason)

        suspicious_results.append({
            "line": line,
            "type": node.get("type"),
            "code": node.get("code"),
            "reason": reason,
            "distance_to_buggy": distance_to_buggy_line(line, buggy_line_location),
            "context": code_context,
            "ast_subtree_json": node.get("ast_subtree_json"),
        })

    # 원본 item을 복사하고 분석 결과를 추가/업데이트
    result = item.copy()
    result.update({
        "buggy_line_context": get_code_context(function_before, buggy_line_location, window=3),
        "suspicious_nodes_topk": suspicious_results,
    })
    return result


# -------------------- CLI 및 실행기 --------------------

def load_model(model_name: str):
    print(f"모델 로딩중: {model_name} (시간이 다소 걸릴 수 있습니다)")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    max_memory_config = {0: "24GiB", "cpu": "90GiB"}  # VRAM 24GB에 맞게 수정
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            max_memory=max_memory_config,
            attn_implementation="flash_attention_2"
        )
    except (ImportError, RuntimeError):
        print("Flash Attention 2를 사용할 수 없거나 호환되지 않습니다. 기본 Attention으로 모델을 로드합니다.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            max_memory=max_memory_config
        )
    model.eval()
    print("모델 로딩 완료.")
    return tokenizer, model


def main():
    # --- ‼️ 모델 경로 고정 (대소문자 구분 주의) ---
    # 로컬에 다운로드한 'Qwen/Qwen1.5-7B' 모델 폴더 경로를 지정합니다.
    LOCAL_MODEL_PATH = "./qwen1.5-7b-chat"

    parser = argparse.ArgumentParser(description="AST와 LLM을 이용한 파이썬 버그 분석기")
    # ‼️ 수정: 기본 경로에서 'Results/' 제거
    parser.add_argument("--input_json", default="./Results/1.DataSeterAgentResult.json", help="분석할 버그 데이터가 담긴 JSON 파일")
    parser.add_argument("--output_json", default="./Results/2.LLM_Analyzed_Results.json", help="분석 결과를 저장할 JSON 파일 경로")
    parser.add_argument("--topk", type=int, default=10, help="각 버그마다 LLM으로 분석할 상위 K개의 의심 노드 수")
    parser.add_argument("--save_every", type=int, default=25, help="N개의 버그를 분석할 때마다 중간 결과를 저장")
    parser.add_argument("--seed", type=int, default=42, help="재현성을 위한 랜덤 시드")
    parser.add_argument("--no_model", action="store_true", help="LLM 호출을 건너뛰고 AST 분석만 수행")
    args = parser.parse_args()

    # 출력 폴더가 없을 경우 생성
    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.input_json, "r", encoding="utf-8") as f:
        bugs_data = json.load(f)

    tokenizer, model, sampling_params = None, None, None
    if not args.no_model:
        if not os.path.isdir(LOCAL_MODEL_PATH):
            print(f"오류: 모델 경로 '{LOCAL_MODEL_PATH}'를 찾을 수 없습니다.")
            print("스크립트 내의 LOCAL_MODEL_PATH 변수를 올바른 경로로 수정하거나, 해당 경로에 모델을 다운로드하세요.")
            return

        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        tokenizer, model = load_model(LOCAL_MODEL_PATH)
        sampling_params = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05}

    all_ids = list(bugs_data.keys())
    results = {}
    tmp_path = args.output_json + ".tmp"
    total = len(all_ids)

    try:
        for idx, bug_id in enumerate(tqdm(all_ids, desc="Analyzing bugs with LLM"), 1):
            item = bugs_data[bug_id]
            try:
                one = analyze_one_bug(
                    str(bug_id), item, model, tokenizer, sampling_params, args.topk
                )
            except torch.cuda.OutOfMemoryError as oom:
                one = {"error": f"CUDA OOM: {oom}", **item}
                torch.cuda.empty_cache()
            except Exception as e:
                one = {"error": str(e), **item}
            results[str(bug_id)] = one

            if args.save_every and (idx % args.save_every == 0):
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"\n[체크포인트] {idx}/{total} 분석 결과를 임시 저장했습니다 -> {tmp_path}")
    finally:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n최종 결과 저장 완료: {args.output_json}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    main()
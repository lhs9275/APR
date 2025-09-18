import json
import os
import re
import argparse
from copy import deepcopy
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm

try:
    from rank_bm25 import BM25Okapi
except ImportError as e:
    raise SystemExit("rank_bm25가 설치되어 있지 않습니다. `pip install rank-bm25` 후 다시 실행하세요.") from e

# 선택적 최적화: NumPy가 있으면 Top-K 추출을 빠르게 처리
try:
    import numpy as _np  # type: ignore
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False

# --------- IO --------- #
def load_all_bugs(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[all_bugs] 파일을 찾을 수 없습니다: {path}\n"
            f"힌트) --all_bugs 경로를 확인하세요."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # dict 또는 list 모두 허용
    if isinstance(data, dict):
        return data
    elif isinstance(data, list):
        out: Dict[str, Dict[str, Any]] = {}
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                # 비정형 항목은 건너뜀
                continue
            # id 우선순위: bug_id > id > enumerate index
            k = str(item.get("bug_id", item.get("id", i)))
            out[k] = item
        return out
    else:
        raise TypeError("all_bugs_meta_data는 dict 또는 list여야 합니다.")


def load_real_bugs(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[real_bugs] 파일을 찾을 수 없습니다: {path}\n"
            f"힌트) --real_bugs 경로를 확인하세요. 예) /mnt/data/LLM_AST_Reason_with_preds.json"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError("real_bugs_by_func는 dict[func_id]->object 형태여야 합니다.")
    return data

# --------- Tokenize --------- #
_token_re = re.compile(r"\w+")

def tokenize(text: str) -> List[str]:
    if not text:
        return []
    # BM25에서 보통 case-insensitive가 유리함
    return _token_re.findall(text.lower())

# --------- BM25 --------- #
def build_bm25(all_bugs: Dict[str, Dict[str, Any]]) -> Tuple[BM25Okapi, List[str], List[List[str]]]:
    bug_id_list: List[str] = []
    tokenized_corpus: List[List[str]] = []

    for k, v in all_bugs.items():
        # 항목별 고유키: bug_id > id > dict key
        doc_id = str(v.get("bug_id", v.get("id", k)))
        line = (v.get("buggy_line_content") or "").strip()
        tokens = tokenize(line)
        bug_id_list.append(doc_id)
        tokenized_corpus.append(tokens)

    if not tokenized_corpus or all(len(t) == 0 for t in tokenized_corpus):
        raise ValueError(
            "BM25 코퍼스를 만들 수 없습니다. all_bugs에 buggy_line_content가 비어 있거나 전처리 후 토큰이 없습니다."
        )

    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, bug_id_list, tokenized_corpus

# --------- Query picker --------- #
def pick_query_from_func_obj(func_obj: Dict[str, Any]) -> Tuple[str, str]:
    """
    우선순위:
      1) buggy_line_content
      2) bug_line.code
      3) code
      4) code_line
      5) suspicious_nodes_topk[0].code
    이후 비었으면 suspicious_nodes_topk 상위 3개를 이어붙여 폴백
    """
    bug_line = func_obj.get("bug_line", {}) if isinstance(func_obj, dict) else {}
    candidates = [
        ("buggy_line_content", func_obj.get("buggy_line_content")),
        ("bug_line.code", (bug_line.get("code") if isinstance(bug_line, dict) else None)),
        ("code", func_obj.get("code")),
        ("code_line", func_obj.get("code_line")),
    ]

    # suspicious_nodes_topk 폴백(단일)
    s_list = func_obj.get("suspicious_nodes_topk")
    if isinstance(s_list, list) and s_list:
        first = s_list[0]
        if isinstance(first, dict):
            candidates.append(("suspicious_nodes_topk[0].code", first.get("code")))

    for name, c in candidates:
        if isinstance(c, str) and c.strip():
            return c.strip(), name

    # 완전 비었으면 suspicious_nodes_topk 상위 3개 code 이어붙이기
    if isinstance(s_list, list) and s_list:
        codes: List[str] = []
        for node in s_list[:3]:
            if isinstance(node, dict):
                code = node.get("code")
                if isinstance(code, str) and code.strip():
                    codes.append(code.strip())
        joined = " ".join(codes).strip()
        if joined:
            return joined, "suspicious_nodes_topk[:3].code_joined"

    return "", "EMPTY"

# --------- Utils --------- #
def get_current_ids(func_id: Any, func_obj: Optional[Dict[str, Any]]) -> set:
    """
    자기 자신 제외를 위해 비교할 가능한 모든 id 후보를 set으로 반환
    포함: func_id, func_obj.id, func_obj.bug_id
    """
    s = {str(func_id)}
    if isinstance(func_obj, dict):
        for key in ("id", "bug_id"):
            v = func_obj.get(key)
            if v is not None:
                s.add(str(v))
    return s

def candidate_id(bug_info: Dict[str, Any], default_id: str) -> str:
    """
    후보 문서의 식별자: bug_id > id > default
    """
    return str(bug_info.get("bug_id", bug_info.get("id", default_id)))

def top_indices_by_score(scores: List[float], top_n: int) -> List[int]:
    """
    점수 내림차순 상위 top_n 인덱스 반환.
    NumPy 있으면 argpartition으로 가속, 없으면 정렬.
    """
    n = len(scores)
    if top_n >= n:
        # 전체 정렬
        return sorted(range(n), key=lambda i: scores[i], reverse=True)

    if _HAS_NUMPY:
        scores_np = _np.array(scores)
        top_idx = _np.argpartition(-scores_np, top_n - 1)[:top_n]
        # 내림차순 정렬 정교화
        return top_idx[_np.argsort(-scores_np[top_idx])].tolist()
    else:
        return sorted(range(n), key=lambda i: scores[i], reverse=True)[:top_n]

# --------- Main --------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_bugs", default="./all_bugs_meta_data.json",
                        help="코퍼스(all bugs) JSON 경로 (dict 또는 list 허용)")
    parser.add_argument("--real_bugs", default="./Results/2.LLM_Analyzed_Results.json",
                        help="실제 버그(by func) JSON 경로 (dict[func_id]->object)")
    parser.add_argument("--out", default="./Results/3.BM25Result.json",
                        help="출력 파일(JSON)")
    parser.add_argument("--topk", type=int, default=10, help="BM25 상위 후보 개수")
    args = parser.parse_args()

    all_bugs = load_all_bugs(args.all_bugs)
    real_bugs_by_func = load_real_bugs(args.real_bugs)

    bm25, bug_id_list, _ = build_bm25(all_bugs)

    final_results: Dict[str, Any] = {}
    it = tqdm(list(real_bugs_by_func.items()), desc="BM25 Top-K Retrieval", disable=False)

    for func_id, func_obj in it:
        out_obj = deepcopy(func_obj if isinstance(func_obj, dict) else {})

        query_text, source = pick_query_from_func_obj(func_obj if isinstance(func_obj, dict) else {})
        tokenized_query = tokenize(query_text)

        if tokenized_query:
            scores = bm25.get_scores(tokenized_query)
        else:
            # 쿼리가 완전히 비었으면 빈 결과
            scores = None

        top_jsons: List[Dict[str, Any]] = []
        if scores is not None:
            # 인덱스 후보 추출
            sorted_indices = top_indices_by_score(scores, max(1, int(args.topk)))

            # 자기 자신 제외 필터 준비
            current_ids = get_current_ids(func_id, func_obj if isinstance(func_obj, dict) else None)

            for idx in sorted_indices:
                bug_key = bug_id_list[idx]
                bug_info = all_bugs.get(bug_key, {})
                cand_id_val = candidate_id(bug_info, bug_key)

                # ✅ 자기 자신(id/bug_id/func_id 일치) 제외
                if cand_id_val in current_ids:
                    continue

                # 파일 메타
                file_name = None
                file_path = None
                file_info = bug_info.get("file")
                if isinstance(file_info, dict):
                    file_name = file_info.get("file_name")
                    file_path = file_info.get("file_path")

                top_jsons.append({
                    "id": cand_id_val,
                    "project_name": bug_info.get("project_name"),
                    "buggy_line_content": bug_info.get("buggy_line_content"),
                    "function": bug_info.get("function"),
                    "file_name": file_name,
                    "file_path": file_path,
                    "score": float(scores[idx]),
                })

                if len(top_jsons) >= args.topk:
                    break

        out_obj["bm25"] = {
            "code_line": query_text,
            "code_line_source": source,
            "topk": args.topk,
            "top": top_jsons,
        }
        final_results[str(func_id)] = out_obj

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print(f"Saved: {args.out}")
    print(f"Funcs: {len(final_results)}")


if __name__ == "__main__":
    main()
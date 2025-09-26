import re
import ast
import json
import textwrap
import random
import copy
import logging
import hashlib
import sys
import difflib  # 유사도 비교를 위해 import
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from argparse import ArgumentParser, Namespace

from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 추가: 리소스 모니터링
import psutil

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ===== 경로 설정 =====
DEFAULT_IN_PATH = Path("./Results/4.GeneratePromResult.json")
DEFAULT_OUT_PATH = Path("Results/patch/5.PatchesResults10.json")

# ===== 정규식 =====
CODE_FENCE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)
DEFCLASS_HEAD_RE = re.compile(r"^\s*(def|class)\s", re.MULTILINE)

# ===== 금지 토큰 패턴 =====
BANNED_PATTERNS = [
    r"\bimport\s",  # no imports
    r"\.\.\.",  # ellipsis placeholder
    r"\braise\s+NotImplementedError\b",
    r"\bprint\s*\(",  # no debugging prints
]
BANNED_TOKENS = [re.compile(p) for p in BANNED_PATTERNS]


def get_parser():
    parser = ArgumentParser(description="LLM을 사용하여 코드 패치를 순차적으로 생성합니다. (개선된 버전 + Token/CPU/RAM/GPU 로깅)")
    parser.add_argument("--model_name", type=str, default="./qwen1.5-7b-chat", help="사용할 모델의 경로 또는 이름")
    parser.add_argument("--in_file", type=Path, default=DEFAULT_IN_PATH, help="프롬프트가 포함된 입력 JSON 파일")
    parser.add_argument("--out_file", type=Path, default=DEFAULT_OUT_PATH, help="생성된 패치를 저장할 출력 JSON 파일")
    parser.add_argument("--batch_size", type=int, default=12, help="추론에 사용할 배치 사이즈")
    parser.add_argument("--num_patches_to_generate", type=int, default=20, help="목표로 하는 고유한 후보 패치의 수")
    parser.add_argument("--target_candidates", type=int, default=10, help="생성된 후보 중 최종 저장할 상위 패치의 수")
    parser.add_argument("--max_generation_multiplier", type=int, default=4,
                        help="무한 루프 방지를 위해 목표 패치 수 대비 최대 몇 배까지 시도할지 결정")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="생성할 최대 토큰 수")

    # --- 랜덤 샘플링 파라미터 ---
    parser.add_argument("--temp_min", type=float, default=0.1, help="랜덤 샘플링에 사용할 Temperature의 최솟값")
    parser.add_argument("--temp_max", type=float, default=1.2, help="랜덤 샘플링에 사용할 Temperature의 최댓값")
    parser.add_argument("--top_p_min", type=float, default=0.7, help="랜덤 샘플링에 사용할 top_p의 최솟값")
    parser.add_argument("--top_p_max", type=float, default=0.95, help="랜덤 샘플링에 사용할 top_p의 최댓값")
    parser.add_argument("--repetition_penalty", type=float, default=1.05, help="반복 페널티")
    parser.add_argument("--seed", type=int, default=42, help="재현성을 위한 랜덤 시드")
    return parser


# ===== 헬퍼 함수 =====
def safe_get(data: Any, path: List[Any], default: Any = None) -> Any:
    """중첩된 딕셔너리나 리스트에서 안전하게 값을 가져옵니다."""
    current = data
    for key in path:
        try:
            current = current[key]
        except (KeyError, IndexError, TypeError):
            return default
    return current


def extract_correct_block(text: str) -> str:
    """LLM 출력에서 코드 블록을 추출합니다."""
    if not text:
        return ""
    # '##correct' 태그 이후의 텍스트를 대상으로 함
    m = re.search(r"^[ \t]*##correct[ \t]*$", text, flags=re.IGNORECASE | re.MULTILINE)
    sub = text[m.end():].strip() if m else text.strip()

    m2 = CODE_FENCE_RE.search(sub)
    candidate = (m2.group(1) if m2 else sub).strip()
    candidate = re.split(r"^###\s", candidate, flags=re.MULTILINE)[0].strip()

    m3 = DEFCLASS_HEAD_RE.search(candidate)
    if m3:
        candidate = candidate[m3.start():]
    candidate = candidate.replace("\x00", "").replace("```", "")
    try:
        if candidate:
            candidate = textwrap.dedent(candidate).strip()
    except IndentationError:
        pass
    return candidate


def ast_ok(py_text: str) -> Tuple[bool, Optional[str]]:
    """코드가 유효한 파이썬 문법인지 AST 파싱으로 확인합니다."""
    try:
        ast.parse(py_text)
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def canon(code: str) -> str:
    """코드에서 주석과 공백을 제거하여 정규화된 형태로 만듭니다."""
    s = re.sub(r"#.*", "", code)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def score_candidate(row: Dict[str, Any], code: str, ast_pass: bool) -> int:
    """페널티 기반의 유연한 점수 평가 함수"""
    try:
        if not code.strip():
            return -1000  # 빈 코드는 최하점

        score = 0

        # 1. AST 통과는 가장 중요한 기준
        if ast_pass:
            score += 100
        else:
            return -500  # 문법 오류는 치명적

        # 2. 원본 함수/클래스 이름이 유지되었는지 확인 (페널티 적용)
        fn_name = (safe_get(row, ["function", "function_name"]) or "").strip()
        if fn_name and re.search(rf"^\s*(def|class)\s+{re.escape(fn_name)}\b", code, re.M):
            score += 25
        else:
            score -= 50  # 함수/클래스 이름이 바뀌면 큰 페널티

        # 3. 원본 코드와의 유사도 (최소 변경 선호)
        original_buggy_code = safe_get(row, ["function", "function_before"], "")
        if original_buggy_code:
            seq_matcher = difflib.SequenceMatcher(None, canon(original_buggy_code), canon(code))
            similarity_score = int(seq_matcher.ratio() * 30)
            score += similarity_score

        # 4. 금지/미구현 패턴에 대한 페널티
        if "pass" in code or "..." in code or "NotImplementedError" in code:
            score -= 50
        for pat in BANNED_TOKENS:
            if pat.search(code):
                score -= 20  # 금지된 패턴(import, print 등)이 있으면 감점

        # 5. 코드 길이에 대한 점수
        num_lines = len(code.strip().split('\n'))
        if num_lines <= 2:
            score -= 30
        elif num_lines > 100:
            score -= 20

        return score
    except Exception:
        return 0


# ===== LLM 클라이언트 =====
class QwenClient:
    def __init__(self, model_name: str):
        logging.info(f"Loading model: {model_name} with 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        logging.info("Model loaded successfully in 4-bit with Flash Attention 2.")
        try:
            self.model = torch.compile(self.model)
            logging.info("Model compiled with torch.compile for faster inference.")
        except Exception as e:
            logging.warning(f"torch.compile failed, proceeding without it. Error: {e}")

        # 리소스 모니터 준비
        self._process = psutil.Process(os.getpid())

    def _gpu_mem_mb(self) -> Optional[float]:
        try:
            if torch.cuda.is_available():
                # 현재 프로세스가 점유 중인 할당 메모리
                mem = torch.cuda.memory_allocated() / (1024 * 1024)
                return float(mem)
        except Exception:
            pass
        return None

    @torch.no_grad()
    def generate(self, prompts: List[str], gen_args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Returns: List[Dict] with keys:
          - text, input_tokens, output_tokens, total_tokens
          - ram_mb, cpu_percent, gpu_mem_mb (optional), elapsed_ms
        """
        all_chats = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            try:
                chat = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                usr_msg = messages[0]["content"]
                chat = f"<|user|>\n{usr_msg}\n<|assistant|>\n"
            all_chats.append(chat)

        # Tokenize with padding/truncation
        inputs = self.tokenizer(
            all_chats,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)
        gen_args["num_return_sequences"] = 1

        # 입력 토큰 수(아이템별) 계산: attention_mask 합계로 패딩 제외한 실제 길이
        attention = inputs.get("attention_mask", None)
        if attention is not None:
            per_input_len = [int(att.sum().item()) for att in attention]
        else:
            # fallback (덜 정확하지만 안전)
            per_input_len = [inputs["input_ids"].shape[1]] * inputs["input_ids"].shape[0]

        # 리소스/시간 측정 시작
        t0 = time.time()

        gen_out = self.model.generate(
            **inputs,
            **gen_args,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        elapsed_ms = (time.time() - t0) * 1000.0

        input_seq_len = inputs["input_ids"].shape[1]
        only_gen_ids = gen_out[:, input_seq_len:]
        decoded_list = self.tokenizer.batch_decode(only_gen_ids, skip_special_tokens=True)

        results: List[Dict[str, Any]] = []
        for i, decoded in enumerate(decoded_list):
            # 출력 토큰 길이(아이템별)
            out_len = int(only_gen_ids[i].shape[0])
            in_len = per_input_len[i]
            total = in_len + out_len

            # RAM/CPU/GPU
            try:
                ram_mb = self._process.memory_info().rss / (1024 * 1024)
            except Exception:
                ram_mb = None
            try:
                # 짧은 샘플링; 너무 길면 추론 속도에 영향
                cpu_percent = self._process.cpu_percent(interval=0.01)
            except Exception:
                cpu_percent = None
            gpu_mb = self._gpu_mem_mb()

            logging.info(
                f"[TokenUsage] Input: {in_len}, Output: {out_len}, Total: {total} | "
                f"RAM: {ram_mb:.2f} MB | CPU: {cpu_percent if cpu_percent is not None else 'NA'}% | "
                f"GPU: {f'{gpu_mb:.2f} MB' if gpu_mb is not None else 'NA'} | "
                f"Elapsed: {elapsed_ms:.1f} ms (batch)"
            )

            results.append({
                "text": decoded,
                "input_tokens": in_len,
                "output_tokens": out_len,
                "total_tokens": total,
                "ram_mb": ram_mb,
                "cpu_percent": cpu_percent,
                "gpu_mem_mb": gpu_mb,
                "elapsed_ms": elapsed_ms,
            })

        return results


def load_with_prompts(path: Path) -> Dict[str, Dict[str, Any]]:
    """다양한 bug_id 키를 처리하도록 강화된 JSON 로더"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    elif isinstance(data, list):
        out: Dict[str, Dict[str, Any]] = {}
        for row in data:
            if not isinstance(row, dict):
                continue
            bid_keys = ["bug_id", "id", "bugsinpy_id"]
            bid = next((str(row.get(key)) for key in bid_keys if row.get(key) is not None), None)
            if bid is None:
                bid = str(len(out))  # Fallback ID
            out[bid] = row
        return out
    else:
        raise ValueError("Unsupported JSON structure")


def post_process(raw_output: Any, temp: float, top_p: float) -> Dict[str, Any]:
    """후처리 함수 (토큰/리소스 메타데이터 유지)"""
    try:
        if isinstance(raw_output, dict):
            raw_text = raw_output.get("text", "")
        else:
            raw_text = str(raw_output)

        # 중국어 포함 시 필터링 (Qwen 모델 특화)
        if re.search(r'[\u4e00-\u9fff]', raw_text):
            base = {
                "raw": raw_text, "code": "", "ast_ok": False, "error": "Chinese characters detected.",
                "temperature": temp, "top_p": top_p
            }
            if isinstance(raw_output, dict):
                base.update({
                    "input_tokens": raw_output.get("input_tokens"),
                    "output_tokens": raw_output.get("output_tokens"),
                    "total_tokens": raw_output.get("total_tokens"),
                    "ram_mb": raw_output.get("ram_mb"),
                    "cpu_percent": raw_output.get("cpu_percent"),
                    "gpu_mem_mb": raw_output.get("gpu_mem_mb"),
                    "elapsed_ms": raw_output.get("elapsed_ms"),
                })
            return base

        code = extract_correct_block(raw_text)
        code = re.split(r"(<\|assistant\|>|<\|user\|>|Here is|Explanation:|Therefore)", code)[0].strip()
        ok, err = ast_ok(code)

        out = {
            "raw": raw_text, "code": code, "ast_ok": ok, "error": err or "",
            "temperature": temp, "top_p": top_p
        }
        if isinstance(raw_output, dict):
            out.update({
                "input_tokens": raw_output.get("input_tokens"),
                "output_tokens": raw_output.get("output_tokens"),
                "total_tokens": raw_output.get("total_tokens"),
                "ram_mb": raw_output.get("ram_mb"),
                "cpu_percent": raw_output.get("cpu_percent"),
                "gpu_mem_mb": raw_output.get("gpu_mem_mb"),
                "elapsed_ms": raw_output.get("elapsed_ms"),
            })
        return out
    except Exception as e:
        return {"raw": "", "code": "", "ast_ok": False, "error": f"Post-processing failed: {e}",
                "temperature": temp, "top_p": top_p}


def _add_candidate_to_state(state: Dict[str, Any], processed_candidate: Dict[str, Any], target_count: int):
    """후보를 상태에 추가하고 완료 여부 업데이트"""
    code = processed_candidate.get("code", "")
    if not code:
        return

    code_hash = hashlib.sha1(canon(code).encode("utf-8")).hexdigest()

    if code_hash not in state["seen_hashes"]:
        state["seen_hashes"].add(code_hash)
        state["candidates"].append(processed_candidate)
        if len(state["candidates"]) >= target_count:
            state["completed"] = True


def process_batch(client: QwenClient, batch: List[Tuple[str, Dict[str, Any]]], args: Namespace):
    batch_states = []
    for bug_id, row in batch:
        prompt = row.get("prompt") or ""
        if prompt and "##correct" not in prompt:
            prompt = prompt.rstrip() + "\n\n##correct\n"

        batch_states.append({
            "bug_id": bug_id, "row": row, "prompt": prompt,
            "candidates": [], "seen_hashes": set(),
            "completed": not bool(prompt),
        })

    # 1. [필수] Greedy Search 실행
    active_states = [s for s in batch_states if not s["completed"]]
    if active_states:
        prompts_to_run = [s["prompt"] for s in active_states]
        raw_outputs = client.generate(prompts_to_run, {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": args.repetition_penalty,
        })
        for state, raw_out in zip(active_states, raw_outputs):
            processed = post_process(raw_out, temp=0.0, top_p=1.0)
            _add_candidate_to_state(state, processed, args.num_patches_to_generate)

    # 2. Random Sampling 루프
    max_attempts = args.num_patches_to_generate * args.max_generation_multiplier
    for _ in range(max_attempts):
        active_states = [s for s in batch_states if not s["completed"]]
        if not active_states:
            logging.info("All items in the batch have completed generation.")
            break

        prompts_to_run = [s["prompt"] for s in active_states]
        temp = random.uniform(args.temp_min, args.temp_max)
        top_p = random.uniform(args.top_p_min, args.top_p_max)

        raw_outputs = client.generate(prompts_to_run, {
            "do_sample": True,
            "max_new_tokens": args.max_new_tokens,
            "temperature": temp,
            "top_p": top_p,
            "repetition_penalty": args.repetition_penalty,
        })

        for state, raw_out in zip(active_states, raw_outputs):
            processed = post_process(raw_out, temp=temp, top_p=top_p)
            _add_candidate_to_state(state, processed, args.num_patches_to_generate)
    else:
        incomplete_ids = [s["bug_id"] for s in batch_states if not s["completed"]]
        if incomplete_ids:
            logging.warning(f"Max attempts reached. Generation may be incomplete for bug IDs: {incomplete_ids}")

    # 3. 최종 결과 정리
    for state in batch_states:
        all_candidates = state["candidates"]
        row = state["row"]
        for cand in all_candidates:
            cand["score"] = score_candidate(row, cand.get("code", ""), cand.get("ast_ok", False))

        all_candidates.sort(key=lambda c: c.get("score", -2000), reverse=True)

        best_choice = all_candidates[0] if all_candidates else {}
        choices = all_candidates[:args.target_candidates]

        # 최종 출력 파일 정리를 위한 불필요한 키 제거
        keys_to_pop = ["prompt", "suspicious_nodes_topk", "bm25", "function", "commit", "description"]
        for key in keys_to_pop:
            row.pop(key, None)

        row["best_choice"] = best_choice
        row["choices"] = choices


def main(args: Namespace):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not args.in_file.exists():
        logging.error(f"Input file not found: {args.in_file}")
        return

    data_dict = load_with_prompts(args.in_file)
    out_dict = copy.deepcopy(data_dict)

    logging.info(f"Targeting {args.num_patches_to_generate} unique patches per bug.")
    logging.info(f"Using batch size: {args.batch_size}")

    client = QwenClient(model_name=args.model_name)
    items = list(out_dict.items())

    with tqdm(total=len(items), desc="Generating patches", ncols=100) as pbar:
        for i in range(0, len(items), args.batch_size):
            batch = items[i:i + args.batch_size]
            try:
                process_batch(client, batch, args)
            except Exception as e:
                bug_ids = [b[0] for b in batch]
                logging.error(f"Error processing batch for bug IDs {bug_ids}: {e}", exc_info=True)
            pbar.update(len(batch))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(out_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info(f"✅ Saved results to: {args.out_file}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
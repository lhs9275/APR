import os
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from argparse import ArgumentParser, Namespace

from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ===== 경로 설정 =====
DEFAULT_IN_PATH = Path("./Results/4.BaseLine.json")
DEFAULT_OUT_PATH = Path("Results/patch/5.BaseLinegemma.json")

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
    parser = ArgumentParser(description="LLM을 사용하여 코드 패치를 순차적으로 생성합니다. (DeepSeek 기본)")
    # ✅ DeepSeek Coder 6.7B Instruct를 기본 모델로 사용
    parser.add_argument("--model_name", type=str,
                        default="google/gemma-7b-it",
                        help="사용할 모델의 경로 또는 Hugging Face repo_id")
    parser.add_argument("--in_file", type=Path, default=DEFAULT_IN_PATH, help="프롬프트가 포함된 입력 JSON 파일")
    parser.add_argument("--out_file", type=Path, default=DEFAULT_OUT_PATH, help="생성된 패치를 저장할 출력 JSON 파일")
    parser.add_argument("--batch_size", type=int, default=4, help="추론에 사용할 배치 사이즈")
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

    # --- 단일 GPU 선택 옵션 ---
    parser.add_argument("--gpu_id", type=int, default=0, help="사용할 단일 GPU ID (예: 0 -> cuda:0)")
    parser.add_argument("--mask_other_gpus", action="store_true",
                        help="True면 CUDA_VISIBLE_DEVICES로 선택한 GPU만 보이게 마스킹")
    return parser


# ===== 헬퍼 함수 =====
def safe_get(data: Any, path: List[Any], default: Any = None) -> Any:
    current = data
    for key in path:
        try:
            current = current[key]
        except (KeyError, IndexError, TypeError):
            return default
    return current


def extract_correct_block(text: str) -> str:
    """LLM 출력에서 ##correct 이후의 코드 블록만 추출"""
    if not text:
        return ""
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
    """AST 파싱으로 문법 검증"""
    try:
        ast.parse(py_text)
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def canon(code: str) -> str:
    """주석/공백 제거한 정규화 문자열"""
    s = re.sub(r"#.*", "", code)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def score_candidate(row: Dict[str, Any], code: str, ast_pass: bool) -> int:
    """페널티 기반 유연 스코어"""
    try:
        if not code.strip():
            return -1000
        score = 0
        if ast_pass:
            score += 100
        else:
            return -500

        fn_name = (safe_get(row, ["function", "function_name"]) or "").strip()
        if fn_name and re.search(rf"^\s*(def|class)\s+{re.escape(fn_name)}\b", code, re.M):
            score += 25
        else:
            score -= 50

        original_buggy_code = safe_get(row, ["function", "function_before"], "")
        if original_buggy_code:
            seq_matcher = difflib.SequenceMatcher(None, canon(original_buggy_code), canon(code))
            similarity_score = int(seq_matcher.ratio() * 30)
            score += similarity_score

        if "pass" in code or "..." in code or "NotImplementedError" in code:
            score -= 50
        for pat in BANNED_TOKENS:
            if pat.search(code):
                score -= 20

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
    """
    이름은 그대로 두었지만 어떤 HF 모델이든 사용 가능.
    - 기본값: deepseek-ai/deepseek-coder-6.7b-instruct
    - 단일 GPU 고정: --gpu_id, --mask_other_gpus
    - 4bit nf4 양자화 + (가능하면) FlashAttention2
    """
    def __init__(self, model_name: str, gpu_id: int = 0, mask_other_gpus: bool = False):
        # (선택) 다른 GPU를 아예 숨기기 (프로세스에서 한 장만 보이게)
        if mask_other_gpus and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            logging.info(f"CUDA_VISIBLE_DEVICES set to {gpu_id} (masking other GPUs).")
            gpu_id = 0  # 마스킹 후엔 보이는 장치가 0번으로 재매핑됨

        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            logging.info(f"Using a single GPU: {self.device}")
        else:
            logging.info("CUDA not available. Using CPU.")

        logging.info(f"Loading model: {model_name} with 4-bit quantization on {self.device}")
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

        # ✅ 단일 장치 고정
        single_device_map = {"": (f"cuda:{gpu_id}" if self.device.type == "cuda" else "cpu")}
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",  # 안 되면 자동으로 fall back 되거나 에러 -> try/except로 교체 가능
            device_map=single_device_map,
        )
        logging.info("Model loaded successfully in 4-bit with Flash Attention 2 on a single device.")
        try:
            self.model = torch.compile(self.model)
            logging.info("Model compiled with torch.compile for faster inference.")
        except Exception as e:
            logging.warning(f"torch.compile failed, proceeding without it. Error: {e}")

    @torch.no_grad()
    def generate(self, prompts: List[str], gen_args: Dict[str, Any]) -> List[str]:
        all_chats = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            try:
                chat = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                usr_msg = messages[0]["content"]
                chat = f"<|user|>\n{usr_msg}\n<|assistant|>\n"
            all_chats.append(chat)

        inputs = self.tokenizer(
            all_chats,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.device)

        gen_args["num_return_sequences"] = 1

        gen_out = self.model.generate(
            **inputs,
            **gen_args,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        input_len = inputs["input_ids"].shape[1]
        only_gen_ids = gen_out[:, input_len:]
        decoded_list = self.tokenizer.batch_decode(only_gen_ids, skip_special_tokens=True)
        return decoded_list


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


def post_process(raw_output: str, temp: float, top_p: float) -> Dict[str, Any]:
    """후처리: 코드 추출 + AST 검증 + 메타 기록"""
    try:
        # 중국어(중문) 포함 시 필터링 (Qwen류 출력 잡음 방지용, DeepSeek에서도 안전)
        if re.search(r'[\u4e00-\u9fff]', raw_output):
            return {"raw": raw_output, "code": "", "ast_ok": False, "error": "Chinese characters detected.",
                    "temperature": temp, "top_p": top_p}

        code = extract_correct_block(raw_output)
        code = re.split(r"(<\|assistant\|>|<\|user\|>|Here is|Explanation:|Therefore)", code)[0].strip()
        ok, err = ast_ok(code)
        return {"raw": raw_output, "code": code, "ast_ok": ok, "error": err or "", "temperature": temp, "top_p": top_p}
    except Exception as e:
        return {"raw": raw_output, "code": "", "ast_ok": False, "error": f"Post-processing failed: {e}",
                "temperature": temp, "top_p": top_p}


def _add_candidate_to_state(state: Dict[str, Any], processed_candidate: Dict[str, Any], target_count: int):
    """후보 수집 & 중복 제거 & 목표치 달성 플래그"""
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

    # 1) Greedy 1샷
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

    # 2) Random Sampling 루프
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

    # 3) 점수화 & 상위 후보 정리
    for state in batch_states:
        all_candidates = state["candidates"]
        row = state["row"]
        for cand in all_candidates:
            cand["score"] = score_candidate(row, cand.get("code", ""), cand.get("ast_ok", False))

        all_candidates.sort(key=lambda c: c.get("score", -2000), reverse=True)

        best_choice = all_candidates[0] if all_candidates else {}
        choices = all_candidates[:args.target_candidates]

        # 출력 경량화를 위한 필드 제거
        keys_to_pop = ["prompt", "suspicious_nodes_topk", "bm25", "function", "commit", "description"]
        for key in keys_to_pop:
            row.pop(key, None)

        row["best_choice"] = best_choice
        row["choices"] = choices


def main(args: Namespace):
    # (선택) 프로세스 시작 직후 마스킹: 외부 스크립트에서 set하지 않아도 됨
    if args.mask_other_gpus and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        logging.info(f"(main) CUDA_VISIBLE_DEVICES set to {args.gpu_id}")

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

    client = QwenClient(model_name=args.model_name,
                        gpu_id=args.gpu_id,
                        mask_other_gpus=args.mask_other_gpus)
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
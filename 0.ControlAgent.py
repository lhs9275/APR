
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import inspect
import io
import json
import logging
import os
import signal
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Protocol
import importlib.util
import subprocess
import uuid

# =========================
# Paths & dynamic importing
# =========================

ROOT = Path(__file__).resolve().parent
# Default relative paths to the user's uploaded files (adjust if your layout differs)
DEFAULT_FILES = {
    "ast": ROOT / "1.LlmASTReasonAnalyzer.py",
    "bugline": ROOT / "2.BugLineFindAgent.py",
    "bm25": ROOT / "3.Bm25.py",
    "prompt": ROOT / "4.GenerateBugfixPrompt.py",
    "patchgen": ROOT / "5.PatchGeneratorAgent.py",
    "eval": ROOT / "6.Eval.py",
}

def load_module_from_path(path: Path):
    """Safely import a Python file even if its name starts with digits."""
    spec = importlib.util.spec_from_file_location(path.stem.replace('.', '_'), path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

# ==========
# Data types
# ==========

@dataclass
class RepoTarget:
    repo_cache_dir: Path
    bug_id: str
    buggy_file: Path
    function_name: Optional[str] = None
    failing_tests: Optional[List[str]] = None  # e.g., ["tests/test_x.py::test_y"]
    test_dir_hint: Optional[Path] = None


@dataclass
class ASTResult:
    suspects: List[Dict[str, Any]] = field(default_factory=list)  # e.g., [{"node_type":"If","lineno":123, ...}]
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BugLineResult:
    candidate_spans: List[Tuple[int, int]] = field(default_factory=list)  # [(start_line, end_line), ...]
    confidence: List[float] = field(default_factory=list)
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    items: List[Dict[str, Any]] = field(default_factory=list)  # e.g., [{"text": "...", "score": 12.3, "meta": {...}}]
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptBundle:
    prompt_texts: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Patch:
    code: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatchSet:
    patches: List[Patch] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    per_candidate: List[Dict[str, Any]] = field(default_factory=list)  # [{"idx":0,"passed":True,"logs":"..."}]
    summary: Dict[str, Any] = field(default_factory=dict)  # {"pass@k":0.34, ...}


# ===============
# Adapter Protocol
# ===============

class AgentAdapter(Protocol):
    async def run(self, **kwargs) -> Any: ...


# -------------------
# Helper: subprocess
# -------------------

async def run_subprocess_json(cmd: List[str], payload: Dict[str, Any], timeout: Optional[int]) -> Any:
    """
    Run a subprocess where the child reads JSON from stdin and writes JSON to stdout.
    Your child script should implement:
        if __name__ == "__main__":
            import json, sys
            data = json.load(sys.stdin)
            ... # do stuff
            json.dump(result_dict, sys.stdout)
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdin_bytes = json.dumps(payload).encode("utf-8")
        stdout, stderr = await asyncio.wait_for(proc.communicate(input=stdin_bytes), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise TimeoutError(f"Subprocess timed out: {' '.join(cmd)}")
    if proc.returncode != 0:
        raise RuntimeError(f"Subprocess failed: {' '.join(cmd)}\nSTDERR:\n{stderr.decode('utf-8', 'ignore')}")
    try:
        return json.loads(stdout.decode("utf-8"))
    except json.JSONDecodeError:
        raise ValueError(f"Subprocess did not return valid JSON. Raw stdout:\n{stdout[:4000]!r}\nSTDERR:\n{stderr[:4000]!r}")

# -------------------
# Adapters (3 modes)
# -------------------

class FunctionAdapter:
    """Call a function reference directly."""
    def __init__(self, fn: Callable[..., Any], timeout: Optional[int] = None):
        self.fn = fn
        self.timeout = timeout

    async def run(self, **kwargs) -> Any:
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(loop.run_in_executor(None, lambda: self.fn(**kwargs)), timeout=self.timeout)


class ClassMethodAdapter:
    """Instantiate a class and call a method."""
    def __init__(self, cls: type, method_name: str = "run", init_kwargs: Optional[Dict[str, Any]] = None, timeout: Optional[int] = None):
        self.cls = cls
        self.method_name = method_name
        self.init_kwargs = init_kwargs or {}
        self.timeout = timeout

    async def run(self, **kwargs) -> Any:
        obj = self.cls(**self.init_kwargs)
        method = getattr(obj, self.method_name)
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(loop.run_in_executor(None, lambda: method(**kwargs)), timeout=self.timeout)


class SubprocessAdapter:
    """Invoke a python script via subprocess with JSON stdin/stdout."""
    def __init__(self, script_path: Path, timeout: Optional[int] = None, python_exe: str = sys.executable):
        self.script_path = str(script_path)
        self.timeout = timeout
        self.python_exe = python_exe

    async def run(self, **kwargs) -> Any:
        cmd = [self.python_exe, "-u", self.script_path]
        return await run_subprocess_json(cmd, kwargs, timeout=self.timeout)

# ==================
# ADAPTERS CONFIG 🔧
# ==================
# Choose ONE of the three modes per agent below and fill in the details.
# By default, we try to import your modules and guess reasonable callables.
# If guessing fails, switch to SubprocessAdapter and implement small __main__ blocks in each file.

def _default_adapter_ast(module):
    # Try class "ASTReasonAnalyzer" with .run(), fallback to function analyze()
    if hasattr(module, "ASTReasonAnalyzer"):
        return ClassMethodAdapter(module.ASTReasonAnalyzer, "run")
    if hasattr(module, "analyze"):
        return FunctionAdapter(module.analyze)
    # Fall back to subprocess if nothing matches
    return SubprocessAdapter(DEFAULT_FILES["ast"], timeout=600)

def _default_adapter_bugline(module):
    if hasattr(module, "BugLineFinder"):
        return ClassMethodAdapter(module.BugLineFinder, "run")
    if hasattr(module, "find_bug_lines"):
        return FunctionAdapter(module.find_bug_lines)
    return SubprocessAdapter(DEFAULT_FILES["bugline"], timeout=600)

def _default_adapter_bm25(module):
    if hasattr(module, "BM25Retriever"):
        return ClassMethodAdapter(module.BM25Retriever, "run")
    if hasattr(module, "retrieve"):
        return FunctionAdapter(module.retrieve)
    return SubprocessAdapter(DEFAULT_FILES["bm25"], timeout=600)

def _default_adapter_prompt(module):
    if hasattr(module, "PromptBuilder"):
        return ClassMethodAdapter(module.PromptBuilder, "run")
    if hasattr(module, "build_prompt"):
        return FunctionAdapter(module.build_prompt)
    return SubprocessAdapter(DEFAULT_FILES["prompt"], timeout=600)

def _default_adapter_patchgen(module):
    if hasattr(module, "PatchGenerator"):
        return ClassMethodAdapter(module.PatchGenerator, "run")
    if hasattr(module, "generate_patches"):
        return FunctionAdapter(module.generate_patches)
    return SubprocessAdapter(DEFAULT_FILES["patchgen"], timeout=1200)

def _default_adapter_eval(module):
    if hasattr(module, "Evaluator"):
        return ClassMethodAdapter(module.Evaluator, "run")
    if hasattr(module, "evaluate"):
        return FunctionAdapter(module.evaluate)
    return SubprocessAdapter(DEFAULTFILES["eval"], timeout=3600)

# =================
# Controller Config
# =================


# ===========
# LLM Router
# ===========
import asyncio, time, hashlib

class LLMRouter:
    """
    Central LLM manager with caching, rate limiting, and routing.
    Agents should call context["llm"].call(prompt, route="patch_gen", **overrides).
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        self.provider = self.cfg.get("provider", "openai")
        self.model = self.cfg.get("model", "gpt-4.1-mini")
        self.endpoint = self.cfg.get("endpoint")
        self.api_key_env = self.cfg.get("api_key_env", "OPENAI_API_KEY")
        self.temperature = self.cfg.get("temperature", 0.2)
        self.max_tokens = self.cfg.get("max_tokens", 2048)
        self.n = self.cfg.get("n", 1)
        self.stop = self.cfg.get("stop")
        self.system_prompt = self.cfg.get("system_prompt", "You are a helpful coding assistant.")
        # simple in-memory cache
        self._cache = {}
        # naive rate limit: N calls per window
        self._last_call = 0.0
        self._min_interval = 1.0 / float(self.cfg.get("rate_limit_rps", 5))

    async def call(self, prompt: str, route: str = "default", **overrides):
        key = hashlib.sha1((route+prompt+str(overrides)).encode("utf-8")).hexdigest()
        if key in self._cache:
            return self._cache[key]
        # naive rate limit
        delta = time.time() - self._last_call
        if delta < self._min_interval:
            await asyncio.sleep(self._min_interval - delta)
        self._last_call = time.time()
        # fake call: In real use, plug OpenAI/vLLM/HF client here
        result = {
            "route": route,
            "model": overrides.get("model", self.model),
            "text": f"[FAKE LLM OUTPUT for route={route}, prompt hash={key[:6]}]",
        }
        self._cache[key] = result
        return result
@dataclass
class ControllerConfig:
    work_dir: Path = ROOT / "workspace"
    llm_config: dict = field(default_factory=dict)

    log_dir: Path = ROOT / "logs"
    top_k: int = 10  # patches per bug
    timeout_ast: int = 300
    timeout_bugline: int = 300
    timeout_bm25: int = 300
    timeout_prompt: int = 300
    timeout_patchgen: int = 1800
    timeout_eval: int = 3600
    retries: int = 1  # per step, on failure
    fail_fast: bool = False
    keep_intermediate: bool = True
    model_routes: dict = field(default_factory=dict)
    llm_max_concurrency: int = 4
    cache_dir: Path = field(default_factory=lambda: ROOT / 'cache')

# ============
# Orchestrator
# ============

class ControllerAgent:
    def __init__(self, cfg: ControllerConfig, module_paths: Dict[str, Path] = DEFAULT_FILES):
        self.cfg = cfg
        self.module_paths = module_paths
        self._setup_dirs()
        self._setup_logging()
        # LLM router
        self.llm = LLMRouter(self.cfg, self.cfg.llm_config)
        # load modules & prepare adapters
        self.adapters = self._init_adapters()

    def _setup_dirs(self):
        self.cfg.work_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.log_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        logfile = self.cfg.log_dir / f"controller_{int(time.time())}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-7s | %(message)s",
            handlers=[logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)],
        )
        logging.info("Logging to %s", logfile)

    def _init_adapters(self) -> Dict[str, AgentAdapter]:
        adapters: Dict[str, AgentAdapter] = {}
        # Try to import modules and choose reasonable defaults
        for key, path in self.module_paths.items():
            if not path.exists():
                logging.warning("Module path missing: %s", path)
        # AST
        adapters["ast"] = _default_adapter_ast(load_module_from_path(self.module_paths["ast"]))
        # BugLine
        adapters["bugline"] = _default_adapter_bugline(load_module_from_path(self.module_paths["bugline"]))
        # BM25
        adapters["bm25"] = _default_adapter_bm25(load_module_from_path(self.module_paths["bm25"]))
        # Prompt
        adapters["prompt"] = _default_adapter_prompt(load_module_from_path(self.module_paths["prompt"]))
        # PatchGen
        adapters["patchgen"] = _default_adapter_patchgen(load_module_from_path(self.module_paths["patchgen"]))
        # Eval
        adapters["eval"] = self._safe_eval_adapter()
        return adapters

    def _safe_eval_adapter(self) -> AgentAdapter:
        try:
            mod = load_module_from_path(self.module_paths["eval"])
            if hasattr(mod, "Evaluator"):
                return ClassMethodAdapter(mod.Evaluator, "run", timeout=self.cfg.timeout_eval)
            if hasattr(mod, "evaluate"):
                return FunctionAdapter(mod.evaluate, timeout=self.cfg.timeout_eval)
            return SubprocessAdapter(self.module_paths["eval"], timeout=self.cfg.timeout_eval)
        except Exception as e:
            logging.warning("Falling back to subprocess adapter for eval due to import error: %s", e)
            return SubprocessAdapter(self.module_paths["eval"], timeout=self.cfg.timeout_eval)

    # ------------------------
    # Retry helper per each op
    # ------------------------
    async def _retry(self, name: str, coro_factory: Callable[[], Any]) -> Any:
        last_exc = None
        for attempt in range(self.cfg.retries + 1):
            try:
                logging.info("[%s] attempt %d", name, attempt + 1)
                return await coro_factory()
            except Exception as e:
                last_exc = e
                logging.exception("[%s] failed on attempt %d: %s", name, attempt + 1, e)
                if attempt < self.cfg.retries:
                    await asyncio.sleep(1.0 * (attempt + 1))
        raise last_exc

    # ---------------
    # Run the pipeline
    # ---------------
    async def run_for_bug(self, target: RepoTarget) -> EvalReport:
        run_id = str(uuid.uuid4())[:8]
        bug_dir = self.cfg.work_dir / f"{target.bug_id}_{run_id}"
        bug_dir.mkdir(parents=True, exist_ok=True)
        logging.info("==> Starting pipeline for bug_id=%s (work: %s)", target.bug_id, bug_dir)

        # Load file content once (kept generic)
        buggy_code = target.buggy_file.read_text(encoding="utf-8")
        context = {
            "repo_cache_dir": str(target.repo_cache_dir),
            "bug_id": target.bug_id,
            "buggy_file": str(target.buggy_file),
            "function_name": target.function_name,
            "failing_tests": target.failing_tests or [],
            "test_dir_hint": str(target.test_dir_hint) if target.test_dir_hint else None,
            "buggy_code": buggy_code,
            "work_dir": str(bug_dir),
            "top_k": self.cfg.top_k,
            "llm_config": self.cfg.llm_config,
            "llm": self.llm,
            "llm": LLMRouter(self.cfg.llm_config),
        }

        # 1) AST
        async def _ast():
            return await self.adapters["ast"].run(**context)
        ast_raw = await self._retry("ASTReasonAnalyzer", _ast)
        ast_res = ASTResult(
            suspects=ast_raw.get("suspects", []),
            notes=ast_raw.get("notes", {}),
        )
        (bug_dir / "1_ast.json").write_text(json.dumps(dataclasses.asdict(ast_res), indent=2), encoding="utf-8")

        # 2) Bug line finder
        async def _bugline():
            payload = {**context, "ast_suspects": ast_res.suspects}
            return await self.adapters["bugline"].run(**payload)
        bl_raw = await self._retry("BugLineFindAgent", _bugline)
        bl_res = BugLineResult(
            candidate_spans=bl_raw.get("candidate_spans", []),
            confidence=bl_raw.get("confidence", []),
            notes=bl_raw.get("notes", {}),
        )
        (bug_dir / "2_buglines.json").write_text(json.dumps(dataclasses.asdict(bl_res), indent=2), encoding="utf-8")

        # 3) Retrieval
        async def _retrieval():
            payload = {**context, "candidate_spans": bl_res.candidate_spans}
            return await self.adapters["bm25"].run(**payload)
        ret_raw = await self._retry("BM25Retriever", _retrieval)
        ret_res = RetrievalResult(
            items=ret_raw.get("items", []),
            notes=ret_raw.get("notes", {}),
        )
        (bug_dir / "3_retrieval.json").write_text(json.dumps(dataclasses.asdict(ret_res), indent=2), encoding="utf-8")

        # 4) Prompt builder
        async def _prompt():
            payload = {
                **context,
                "candidate_spans": bl_res.candidate_spans,
                "retrieved": ret_res.items,
                "ast_suspects": ast_res.suspects,
            }
            return await self.adapters["prompt"].run(**payload)
        pb_raw = await self._retry("PromptBuilder", _prompt)
        pb = PromptBundle(
            prompt_texts=pb_raw.get("prompts", []) or pb_raw.get("prompt_texts", []),
            meta=pb_raw.get("meta", {}),
        )
        (bug_dir / "4_prompts.json").write_text(json.dumps(dataclasses.asdict(pb), indent=2), encoding="utf-8")

        # 5) Patch generation
        async def _patchgen():
            payload = {
                **context,
                "prompts": pb.prompt_texts,
                "candidate_spans": bl_res.candidate_spans,
            }
            return await self.adapters["patchgen"].run(**payload)
        pg_raw = await self._retry("PatchGeneratorAgent", _patchgen)
        patches_list = pg_raw.get("patches", [])
        patch_objs = [Patch(code=p.get("code", ""), meta={k:v for k,v in p.items() if k != "code"}) for p in patches_list]
        patchset = PatchSet(patches=patch_objs, meta=pg_raw.get("meta", {}))
        (bug_dir / "5_patches.json").write_text(json.dumps(dataclasses.asdict(patchset), indent=2), encoding="utf-8")

        # 6) Evaluation
        async def _eval():
            payload = {
                **context,
                "patches": [p.code for p in patchset.patches],
                "patch_meta": [p.meta for p in patchset.patches],
            }
            return await self.adapters["eval"].run(**payload)
        ev_raw = await self._retry("Evaluator", _eval)
        report = EvalReport(
            per_candidate=ev_raw.get("per_candidate", []),
            summary=ev_raw.get("summary", {}),
        )
        (bug_dir / "6_eval.json").write_text(json.dumps(dataclasses.asdict(report), indent=2), encoding="utf-8")

        logging.info("✅ Completed bug_id=%s | summary=%s", target.bug_id, json.dumps(report.summary, ensure_ascii=False))
        return report


def _load_llm_config(arg_value: Optional[str]) -> dict:
    """
    Load LLM config from (1) --llm-config JSON string or file path,
    or (2) LLM_CONFIG env var containing JSON. Returns {} if missing.
    Example keys you might use in your agents:
      {
        "provider": "openai|vllm|transformers",
        "model": "gpt-4.1-mini",
        "endpoint": "http://localhost:8000/v1",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": 0.2,
        "max_tokens": 2048,
        "n": 10,
        "stop": ["```"],
        "system_prompt": "You are a code-fixing model..."
      }
    """
    import os, json
    if arg_value:
        # Try path first
        from pathlib import Path
        pp = Path(arg_value)
        if pp.exists():
            return json.loads(pp.read_text(encoding="utf-8"))
        # else treat as JSON string
        return json.loads(arg_value)
    env_val = os.environ.get("LLM_CONFIG")
    if env_val:
        try:
            return json.loads(env_val)
        except Exception:
            pass
    return {}
# ===========
# CLI utility
# ===========

def parse_args():
    ap = argparse.ArgumentParser(description="ControllerAgent — orchestrate multi-agent bugfix pipeline")
    ap.add_argument("--repo-cache", type=Path, required=True, help="Path to repo cache root")
    ap.add_argument("--bug-id", type=str, required=True, help="Bug identifier, e.g., pandas-123")
    ap.add_argument("--buggy-file", type=Path, required=True, help="Path to the buggy source file")
    ap.add_argument("--function-name", type=str, default=None, help="(optional) target function name")
    ap.add_argument("--tests", type=str, nargs="*", default=None, help="(optional) specific tests to run")
    ap.add_argument("--test-dir-hint", type=Path, default=None, help="(optional) tests dir")
    ap.add_argument("--top-k", type=int, default=10, help="number of candidate patches to generate")
    ap.add_argument("--work-dir", type=Path, default=ROOT/"workspace")
    ap.add_argument("--log-dir", type=Path, default=ROOT/"logs")
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--llm-config", type=str, default=None, help="JSON string or path to JSON with LLM settings")
    return ap.parse_args()

async def _amain(args):
    cfg = ControllerConfig(
        work_dir=args.work_dir,
        log_dir=args.log_dir,
        top_k=args.top_k,
        retries=args.retries,
        fail_fast=args.fail_fast,
        keep_intermediate=args.keep_intermediate,
        llm_config=_load_llm_config(args.llm_config),
        model_routes={},
    )
    controller = ControllerAgent(cfg)
    target = RepoTarget(
        repo_cache_dir=args.repo_cache,
        bug_id=args.bug_id,
        buggy_file=args.buggy_file,
        function_name=args.function_name,
        failing_tests=args.tests,
        test_dir_hint=args.test_dir_hint,
    )
    report = await controller.run_for_bug(target)
    print(json.dumps(dataclasses.asdict(report), ensure_ascii=False))

def main():
    args = parse_args()
    asyncio.run(_amain(args))

if __name__ == "__main__":
    main()

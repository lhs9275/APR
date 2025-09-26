#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, sys, glob
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
import pandas as pd
import matplotlib.pyplot as plt

# Pass@k 성능(본문 수치 고정)
PASSK: Dict[str, Dict[str, float]] = {
    "Qwen-ZeroShot":      {"pass@1": 31.37, "pass@5": 32.90, "pass@10": 33.33},
    "Gemma-ZeroShot":     {"pass@1": 20.78, "pass@5": 25.05, "pass@10": 29.41},
    "DeepSeek-ZeroShot":  {"pass@1": 29.61, "pass@5": 34.86, "pass@10": 39.22},
    "AgentRepair":        {"pass@1": 36.86, "pass@5": 46.68, "pass@10": 50.98},
}

def load_json_any(p: Path) -> Union[Dict[str, Any], List[Any]]:
    with p.open("r", encoding="utf-8") as f:
        s = f.read().strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]

def to_records(model: str, obj: Union[Dict, List]) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    it = (obj.values() if isinstance(obj, dict) else obj) if isinstance(obj, (dict, list)) else []
    for item in it:
        if not isinstance(item, dict):
            continue
        bug_id = item.get("bug_id") or item.get("bugsinpy_id") or item.get("id")

        def push(kind: str, d: Dict[str, Any]) -> None:
            recs.append({
                "Model": model,
                "bug_id": bug_id,
                "kind": kind,
                "input_tokens": d.get("input_tokens"),
                "output_tokens": d.get("output_tokens"),
                "total_tokens": d.get("total_tokens"),
                "ram_mb": d.get("ram_mb"),
                "gpu_mem_mb": d.get("gpu_mem_mb"),
                "elapsed_ms": d.get("elapsed_ms"),
                "temperature": d.get("temperature"),
                "top_p": d.get("top_p"),
                "score": d.get("score"),
                "error": d.get("error"),
                "ast_ok": d.get("ast_ok"),
            })

        best = item.get("best_choice")
        if isinstance(best, dict):
            push("best_choice", best)

        choices = item.get("choices") or []
        for i, ch in enumerate(choices):
            if isinstance(ch, dict):
                push("choice_%d" % i, ch)

    # 최소 하나의 비용 지표가 있는 레코드만 유지
    keep_cols = ["input_tokens", "output_tokens", "total_tokens", "ram_mb", "gpu_mem_mb", "elapsed_ms"]
    return [r for r in recs if any(r.get(k) is not None for k in keep_cols)]

def find_one(explicit: Optional[str], patterns: List[str]) -> Optional[Path]:
    # 1) 명시 경로 우선
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.exists() else None
    # 2) 현재/상위 폴더 + 재귀 검색
    roots = [Path.cwd(), Path.cwd().parent]
    for root in roots:
        for pat in patterns:
            hits = list(root.glob(pat))
            if hits:
                return hits[0]
            hits = glob.glob(str(root / "**" / pat), recursive=True)
            if hits:
                return Path(hits[0])
    return None

def main() -> None:
    ap = argparse.ArgumentParser(description="4-way cost/efficiency plots (AgentRepair vs Qwen/Gemma/DeepSeek)")
    ap.add_argument("--qwen", help="path to 5.BaseLineQwen.json")
    ap.add_argument("--deepseek", help="path to 5.BaseLineDeepSeek.json")
    ap.add_argument("--gemma", help="path to 5.BaseLineGemma.json")
    ap.add_argument("--agent", help="path to 5.PatchesResults10.json")
    ap.add_argument("--outdir", default=".", help="output directory")
    args = ap.parse_args()

    paths: Dict[str, Optional[Path]] = {
        "Qwen-ZeroShot":     find_one(args.qwen,    ["5.BaseLineQwen.json","BaseLineQwen*.json","*Qwen*.json"]),
        "DeepSeek-ZeroShot": find_one(args.deepseek,["5.BaseLineDeepSeek.json","BaseLineDeepSeek*.json","*DeepSeek*.json"]),
        "Gemma-ZeroShot":    find_one(args.gemma,   ["5.BaseLineGemma.json","BaseLineGemma*.json","*Gemma*.json"]),
        "AgentRepair":       find_one(args.agent,   ["5.PatchesResults10.json","*PatchesResults*.json","*AgentRepair*.json"]),
    }

    avail = {m: p for m, p in paths.items() if p is not None}
    missing = [m for m, p in paths.items() if p is None]
    if len(avail) < 2:
        print("❌ Not enough files. Missing:", missing)
        sys.exit(1)

    print("✔ Using files:")
    for m, p in avail.items():
        print("  %-18s -> %s" % (m, p))
    if missing:
        print("⚠ Missing (skipped in plots):", missing)

    # Load & normalize
    all_recs: List[Dict[str, Any]] = []
    for m, p in avail.items():
        data = load_json_any(p)  # type: ignore
        all_recs += to_records(m, data)

    if not all_recs:
        print("No records parsed; check JSON structure.")
        sys.exit(1)

    df = pd.DataFrame(all_recs)
    for c in ["input_tokens","output_tokens","total_tokens","ram_mb","gpu_mem_mb","elapsed_ms"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "elapsed_ms" in df.columns:
        df["elapsed_sec"] = df["elapsed_ms"] / 1000.0
    if "total_tokens" in df.columns:
        mask = df["total_tokens"].isna()
        df.loc[mask, "total_tokens"] = df.loc[mask, ["input_tokens", "output_tokens"]].sum(axis=1, min_count=1)

    agg = {
        "bug_id": pd.Series.nunique,
        "input_tokens": ["mean"],
        "output_tokens": ["mean"],
        "total_tokens": ["mean"],
        "elapsed_sec": ["mean", "median"],
        "ram_mb": ["mean"],
        "gpu_mem_mb": ["mean", "max"],
    }
    summary = df.groupby("Model").agg(agg)
    summary.columns = ["_".join([x for x in t if x]) for t in summary.columns.to_flat_index()]
    summary = summary.rename(columns={"bug_id_nunique": "bugs_covered"}).reset_index()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outdir / "agentrepair_cost_summary.csv", index=False)
    df.to_csv(outdir / "agentrepair_raw_records.csv", index=False)

    # ---- Plots ----
    # 1) 평균 토큰 비용
    ax = summary.set_index("Model")[["input_tokens_mean","output_tokens_mean","total_tokens_mean"]].plot(kind="bar")
    ax.set_ylabel("Tokens (avg per candidate)")
    ax.set_title("Average Token Cost per Candidate")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)  # ✅ 이름 가로
    ax.figure.tight_layout()
    ax.figure.savefig(outdir / "avg_tokens.png", dpi=150)
    plt.close(ax.figure)

    # 2) 효율성 곡선: Pass@k vs 토큰 비용
    eff = summary[["Model","total_tokens_mean"]].copy()
    for m in eff["Model"]:
        if m in PASSK:
            for k, v in PASSK[m].items():
                eff.loc[eff["Model"] == m, k] = v
    fig, ax = plt.subplots()
    for _, r in eff.iterrows():
        xs = [r["total_tokens_mean"]] * 3
        ys = [r["pass@1"], r["pass@5"], r["pass@10"]]
        ax.plot(xs, ys, marker="o", label=r["Model"])
    ax.set_xlabel("Avg Total Tokens per Candidate")
    ax.set_ylabel("Pass@k (%)")
    ax.set_title("Efficiency Curve: Pass@k vs Token Cost")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "efficiency_curve.png", dpi=150)
    plt.close(fig)

    # 3) VRAM (avg vs max)
    ax = summary.set_index("Model")[["gpu_mem_mb_mean","gpu_mem_mb_max"]].plot(kind="bar")
    ax.set_ylabel("VRAM (MB)")
    ax.set_title("VRAM Usage Profile (avg vs max)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)  # ✅ 이름 가로
    ax.figure.tight_layout()
    ax.figure.savefig(outdir / "vram_profile.png", dpi=150)
    plt.close(ax.figure)

    # 4) Runtime (avg vs median)
    ax = summary.set_index("Model")[["elapsed_sec_mean","elapsed_sec_median"]].plot(kind="bar")
    ax.set_ylabel("Seconds")
    ax.set_title("Runtime Profile (avg vs median)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)  # ✅ 이름 가로
    ax.figure.tight_layout()
    ax.figure.savefig(outdir / "time_profile.png", dpi=150)
    plt.close(ax.figure)

    print("\n✅ Saved to", str(outdir.resolve()))
    print("   - agentrepair_cost_summary.csv / agentrepair_raw_records.csv")
    print("   - avg_tokens.png, efficiency_curve.png, vram_profile.png, time_profile.png")

if __name__ == "__main__":
    main()
# 파일명: tokens_report.py  (← token.py, reporttk.py 말고!)
import json, math, random
import numpy as np
from pathlib import Path

RESULT_PATH = Path("../Results/patch/5.PatchesResults10.json")

def safe_vals(seq):
    out = []
    for v in seq:
        if v is None:
            continue
        try:
            if math.isnan(v):
                continue
        except TypeError:
            pass
        out.append(float(v))
    return out

def mean_std(vals):
    vals = safe_vals(vals)
    if not vals:
        return None, None
    arr = np.array(vals, dtype=float)
    mean = float(arr.mean())
    std  = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mean, std

def bootstrap_ci(vals, metric=np.mean, iters=2000, alpha=0.05):
    vals = safe_vals(vals)
    if not vals:
        return None
    arr = np.array(vals, dtype=float)
    if len(arr) == 1:
        m = float(metric(arr))
        return (m, m)
    n = len(arr)
    boots = []
    for _ in range(iters):
        sample = np.random.choice(arr, size=n, replace=True)
        boots.append(metric(sample))
    boots = np.sort(np.array(boots))
    lo = float(np.percentile(boots, 100*(alpha/2)))
    hi = float(np.percentile(boots, 100*(1-alpha/2)))
    return (lo, hi)

def main():
    data = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    rows = []
    for bug_id, row in data.items():
        best = row.get("best_choice", {}) or {}
        rows.append({
            "bug_id": bug_id,
            "success": bool(best.get("ast_ok", False) and (best.get("code") or "").strip()),
            "input_tokens": best.get("input_tokens"),
            "output_tokens": best.get("output_tokens"),
            "total_tokens": best.get("total_tokens"),
            "ram_mb": best.get("ram_mb"),
            "cpu_percent": best.get("cpu_percent"),
            "gpu_mem_mb": best.get("gpu_mem_mb"),
            "elapsed_ms": best.get("elapsed_ms"),
        })

    tot_tokens = [r["total_tokens"] for r in rows]
    ram = [r["ram_mb"] for r in rows]
    cpu = [r["cpu_percent"] for r in rows]
    gpu = [r["gpu_mem_mb"] for r in rows]
    t_ms = [r["elapsed_ms"] for r in rows]

    m_tok, s_tok = mean_std(tot_tokens)
    m_ram, s_ram = mean_std(ram)
    m_cpu, s_cpu = mean_std(cpu)
    m_gpu, s_gpu = mean_std(gpu)
    m_ms,  s_ms  = mean_std(t_ms)

    tok_ci = bootstrap_ci(tot_tokens)
    ram_ci = bootstrap_ci(ram)
    ms_ci  = bootstrap_ci(t_ms)

    fixed = sum(1 for r in rows if r["success"])
    tok_sum = sum(safe_vals(tot_tokens)) or 1.0
    fixes_per_million = fixed / (tok_sum / 1e6)

    print("=== AgentRepair Efficiency Report ===")
    print(f"Num bugs: {len(rows)}  |  Fixed: {fixed}")
    print(f"Avg total tokens / bug: {m_tok:.2f} ± {s_tok:.2f}" if m_tok is not None else "Avg total tokens / bug: N/A")
    if tok_ci: print(f"  95% CI: {tok_ci[0]:.2f}–{tok_ci[1]:.2f}")
    print(f"Avg RAM (MB) / bug: {m_ram:.2f} ± {s_ram:.2f}" if m_ram is not None else "Avg RAM (MB) / bug: N/A")
    if ram_ci: print(f"  95% CI: {ram_ci[0]:.2f}–{ram_ci[1]:.2f}")
    print(f"Avg CPU (%) / bug: {m_cpu:.2f} ± {s_cpu:.2f}" if m_cpu is not None else "Avg CPU (%) / bug: N/A")
    print(f"Avg GPU mem (MB) / bug: {m_gpu:.2f} ± {s_gpu:.2f}" if m_gpu is not None else "Avg GPU mem (MB) / bug: N/A")
    print(f"Avg time (ms) / bug: {m_ms:.2f} ± {s_ms:.2f}" if m_ms is not None else "Avg time (ms) / bug: N/A")
    if ms_ci: print(f"  95% CI: {ms_ci[0]:.2f}–{ms_ci[1]:.2f}")
    print(f"Fixes per 1M tokens: {fixes_per_million:.3f}")

if __name__ == "__main__":
    np.random.seed(42); random.seed(42)
    main()
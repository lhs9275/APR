# -*- coding: utf-8 -*-
"""
AgentRepair vs HAFix 시각화 묶음 (matplotlib-only)
- 벤다이어그램(유사) : bug-level success
- pass@k 막대 + HAFix 범위(세로선)
- 모델×휴리스틱 히트맵(버그 성공 개수) + 요약 막대
- pass@k 히트맵 + 요약 막대

실행하면 /mnt/data 아래에 PNG 파일이 생성됩니다.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import itertools

# -----------------------------
# 0) 데이터 입력 (필요시 여기만 수정)
# -----------------------------
TOTAL_BUGS = 51

# AgentRepair 결과
AGENTREPAIR_BUG_SUCC = 23
AGENTREPAIR_PASSK = (36.0, 43.5, 45.1)  # (p1,p5,p10)

# Zero-shot 결과
ZEROSHOT_PASSK = (30.1, 37.6, 39.2)

# HAFix: bug-level success (51개 중) - 모델×휴리스틱
HAFIX_BUG_SUCCESS = {
    "CodeLlama-7B": {
        "Baseline": 24, "CFN-modified": 25, "CFN-all": 24, "FN-modified": 22,
        "FN-all": 22, "FLN-all": 24, "FN-pair": 12, "FL-diff": 13
    },
    "DeepSeek-6.7B": {
        "Baseline": 23, "CFN-modified": 22, "CFN-all": 25, "FN-modified": 24,
        "FN-all": 25, "FLN-all": 24, "FN-pair": 16, "FL-diff": 19
    },
    "DeepSeekV2-16B": {
        "Baseline": 18, "CFN-modified": 19, "CFN-all": 18, "FN-modified": 18,
        "FN-all": 21, "FLN-all": 18, "FN-pair": 18, "FL-diff": 17
    }
}

# HAFix: pass@k 표 (모델×휴리스틱)
# 각 값은 (%)로 기입
HAFIX_PASSK = {
    "CodeLlama-7B": {
        "CFN-modified": (22.35, 39.17, 49.02),
        "CFN-all":      (22.16, 39.15, 47.06),
        "FN-modified":  (20.78, 35.68, 43.14),
        "FN-all":       (21.37, 35.45, 43.14),
        "FLN-all":      (24.90, 39.65, 47.06),
        "FN-pair":      (13.53, 19.44, 23.53),
        "FL-diff":      (12.94, 20.92, 25.49),
    },
    "DeepSeek-6.7B": {
        "CFN-modified": (32.16, 40.41, 43.14),
        "CFN-all":      (31.37, 44.00, 49.02),
        "FN-modified":  (27.06, 42.53, 47.06),
        "FN-all":       (25.10, 41.52, 49.02),
        "FLN-all":      (30.78, 43.58, 47.06),
        "FN-pair":      (15.29, 24.88, 31.37),
        "FL-diff":      (19.22, 31.79, 37.25),
    },
    "DeepSeekV2-16B": {
        "CFN-modified": (26.08, 33.00, 37.25),
        "CFN-all":      (26.47, 31.87, 35.29),
        "FN-modified":  (29.41, 34.10, 35.29),
        "FN-all":       (29.61, 37.08, 41.18),
        "FLN-all":      (27.65, 32.46, 35.29),
        "FN-pair":      (21.37, 31.87, 35.29),
        "FL-diff":      (24.51, 31.43, 33.33),
    }
}

OUT_DIR = "/mnt/data"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------
# 1) Bug-level Success 벤다이어그램(유사)
# ------------------------------------------------
def make_venn_like(agent_succ, hafix_best, total_bugs, outpath):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect('equal')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    c1 = Circle((4,5), 3.0, fill=False, linewidth=2)
    c2 = Circle((6,5), 3.0, fill=False, linewidth=2)
    ax.add_patch(c1)
    ax.add_patch(c2)

    ax.text(3, 8.8, f"Bug-level Success (BugsInPy, N={total_bugs})", fontsize=12, ha='center')
    ax.text(3.2, 5.0, f"AgentRepair\n{agent_succ}/{total_bugs}", ha='center', va='center', fontsize=12)
    ax.text(6.8, 5.0, f"HAFix (best config)\n{hafix_best}/{total_bugs}", ha='center', va='center', fontsize=12)
    ax.text(5.0, 5.0, "Overlap\nunknown", ha='center', va='center', fontsize=10)
    ax.text(5.0, 0.8, "Note: exact overlap needs per-bug success lists.", ha='center', fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

# HAFix best bug success
hafix_best_bug_succ = max(max(d.values()) for d in HAFIX_BUG_SUCCESS.values())
make_venn_like(AGENTREPAIR_BUG_SUCC, hafix_best_bug_succ, TOTAL_BUGS,
               os.path.join(OUT_DIR, "agentrepair_hafix_venn_like.png"))

# ------------------------------------------------
# 2) pass@k 비교 막대 + HAFix 범위(세로선)
# ------------------------------------------------
def extract_hafix_ranges(passk_dict):
    # 각 k에 대해 전체 모델×휴리스틱 값의 min/max 추출
    p1_vals, p5_vals, p10_vals = [], [], []
    for model, heur_map in passk_dict.items():
        for heur, (p1,p5,p10) in heur_map.items():
            p1_vals.append(p1); p5_vals.append(p5); p10_vals.append(p10)
    return (min(p1_vals), max(p1_vals)), (min(p5_vals), max(p5_vals)), (min(p10_vals), max(p10_vals))

def make_passk_bar_with_ranges(agent_p, zero_p, hafix_ranges, outpath):
    labels = ['pass@1','pass@5','pass@10']
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7,5))
    ax.bar(x - width, [agent_p[0], agent_p[1], agent_p[2]], width, label='AgentRepair')
    ax.bar(x,         [zero_p[0],  zero_p[1],  zero_p[2]],  width, label='LLM Zero-shot')

    # HAFix ranges (thick line + midpoint marker)
    for i, k in enumerate(['p1','p5','p10']):
        mn, mx = {'p1':hafix_ranges[0], 'p5':hafix_ranges[1], 'p10':hafix_ranges[2]}[k]
        ax.vlines(x[i] + width, mn, mx, linewidth=6, label=None)
        ax.plot([x[i] + width], [(mn+mx)/2], marker='o')

        ax.text(x[i] + width, mx + 1, f"{mn:.1f}–{mx:.1f}", ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x, labels)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('BugsInPy: AgentRepair vs Zero-shot vs HAFix (range over heuristics/models)')
    ax.legend(loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # annotate exact bars
    for i, val in enumerate([agent_p[0], agent_p[1], agent_p[2]]):
        ax.text(x[i] - width, val + 1, f"{val:.1f}", ha='center', va='bottom', fontsize=9)
    for i, val in enumerate([zero_p[0], zero_p[1], zero_p[2]]):
        ax.text(x[i], val + 1, f"{val:.1f}", ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

hafix_ranges = extract_hafix_ranges(HAFIX_PASSK)
make_passk_bar_with_ranges(AGENTREPAIR_PASSK, ZEROSHOT_PASSK, hafix_ranges,
                           os.path.join(OUT_DIR, "agentrepair_hafix_passk_comparison.png"))

# ------------------------------------------------
# 3) 모델×휴리스틱 히트맵 (bug-level success) + 요약 막대
# ------------------------------------------------
def to_matrix_bug_success(hafix_bug):
    models = list(hafix_bug.keys())
    heuristics = list(next(iter(hafix_bug.values())).keys())
    mat = np.zeros((len(models), len(heuristics)), dtype=float)
    for i, m in enumerate(models):
        for j, h in enumerate(heuristics):
            mat[i, j] = hafix_bug[m][h]
    return models, heuristics, mat

def draw_heatmap(values, xticklabels, yticklabels, title, cbar_label, outpath):
    fig, ax = plt.subplots(figsize=(10,6))
    im = ax.imshow(values, aspect='auto')
    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_yticks(np.arange(len(yticklabels)))
    ax.set_xticklabels(xticklabels, rotation=45, ha='right')
    ax.set_yticklabels(yticklabels)
    ax.set_title(title)

    # annotate cells
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.0f}", ha='center', va='center', color='white' if values[i,j] > values.max()/2 else 'black')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

models, heuristics, bug_mat = to_matrix_bug_success(HAFIX_BUG_SUCCESS)
draw_heatmap(
    bug_mat,
    heuristics,
    models,
    "Bug-level Success: HAFix (models × heuristics)",
    "Success count (out of 51)",
    os.path.join(OUT_DIR, "bugsuccess_heatmap.png")
)

# 요약 막대 (AgentRepair vs HAFix best/avg)
hafix_best_overall = bug_mat.max()
hafix_avg_overall = bug_mat.mean()
def bar_simple(labels, heights, title, ylabel, outpath):
    fig, ax = plt.subplots(figsize=(6,5))
    bars = ax.bar(labels, heights)
    for b, h in zip(bars, heights):
        ax.text(b.get_x() + b.get_width()/2, h + 0.5, f"{h:.1f}", ha='center', va='bottom')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

bar_simple(
    ["AgentRepair", "HAFix (best)", "HAFix (avg)"],
    [AGENTREPAIR_BUG_SUCC, hafix_best_overall, hafix_avg_overall],
    "AgentRepair vs HAFix (Bug-level Success Summary)",
    "Success count (out of 51)",
    os.path.join(OUT_DIR, "bugsuccess_bar.png")
)

# ------------------------------------------------
# 4) pass@k 히트맵 + 요약 막대
# ------------------------------------------------
def to_matrix_passk(hafix_passk, metric_index):
    """
    metric_index: 0=pass@1, 1=pass@5, 2=pass@10
    """
    models = list(hafix_passk.keys())
    heuristics = list(next(iter(hafix_passk.values())).keys())
    mat = np.zeros((len(models), len(heuristics)), dtype=float)
    for i, m in enumerate(models):
        for j, h in enumerate(heuristics):
            mat[i, j] = hafix_passk[m][h][metric_index]
    return models, heuristics, mat

for idx, name in enumerate(["pass@1", "pass@5", "pass@10"]):
    m, h, mat = to_matrix_passk(HAFIX_PASSK, idx)
    draw_heatmap(
        mat, h, m,
        f"{name} Heatmap: HAFix (models × heuristics)",
        f"{name} (%)",
        os.path.join(OUT_DIR, f"passk_heatmap_{name.replace('@','at')}.png")
    )

    # 요약 막대: AgentRepair vs Zero-shot vs HAFix(best/avg)
    hafix_best = mat.max()
    hafix_avg = mat.mean()
    agent_val = AGENTREPAIR_PASSK[idx]
    zero_val = ZEROSHOT_PASSK[idx]
    bar_simple(
        ["AgentRepair", "Zero-shot", "HAFix (best)", "HAFix (avg)"],
        [agent_val, zero_val, hafix_best, hafix_avg],
        f"{name} Summary: AgentRepair vs Zero-shot vs HAFix",
        f"{name} (%)",
        os.path.join(OUT_DIR, f"passk_bar_{name.replace('@','at')}.png")
    )

print("Done. Files saved to", OUT_DIR)
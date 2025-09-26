# fig4_passk_beauty.py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ===== 폰트 (Times가 있으면 사용, 없으면 serif fallback) =====
# mpl.rcParams['font.family'] = 'Times New Roman'  # 로컬에 있으면 주석 해제
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9

# ===== 데이터 (실측 Pass@k, %) =====
k = np.arange(1, 11)

Qwen     = [31.37, 32.07, 32.42, 32.68, 32.90, 33.07, 33.20, 33.29, 33.33, 33.33]
Gemma    = [20.78, 21.92, 23.01, 24.05, 25.05, 26.01, 26.93, 27.80, 28.63, 29.41]
DeepSeek = [29.61, 31.46, 32.75, 33.85, 34.86, 35.82, 36.73, 37.60, 38.43, 39.22]
AgentRep = [36.86, 40.74, 43.25, 45.15, 46.68, 47.91, 48.91, 49.72, 50.39, 50.98]

# ========= 스타일 프리셋 =========
# 1) "clean"       : 컬러블라인드 세이프, 얇은 라인, 그리드 최소화
# 2) "mono"        : 흑백 인쇄용, 라인/마커/라인스타일만으로 구분
# 3) "editorial"   : 잡지/포스터 감성, 여백 여유, AgentRepair 강조

THEME = "clean"   # ← 여기서 바꿔가며 저장: "clean" / "mono" / "editorial"

def get_style(theme):
    if theme == "clean":
        # Colorblind-safe 팔레트
        colors = {
            "Qwen-ZeroShot":     "#1b9e77",
            "Gemma-ZeroShot":    "#d95f02",
            "DeepSeek-ZeroShot": "#7570b3",
            "AgentRepair":       "#e7298a",
        }
        styles = {
            "Qwen-ZeroShot":     dict(color="#D1A400", marker='o',  linewidth=1.9, markersize=5),
            "Gemma-ZeroShot":    dict(color="#00D13B", marker='s',  linewidth=1.9, markersize=5),
            "DeepSeek-ZeroShot": dict(color="#0012D1", marker='^',  linewidth=1.9, markersize=5),
            "AgentRepair":       dict(color="#d10000", marker='D',  linewidth=2.5, markersize=6),
        }
        grid = dict(linestyle="--", linewidth=0.7, alpha=0.35)
        return styles, grid, False


    raise ValueError("Unknown THEME")

series_data = [
    ("Qwen-ZeroShot",     Qwen),
    ("Gemma-ZeroShot",    Gemma),
    ("DeepSeek-ZeroShot", DeepSeek),
    ("AgentRepair",       AgentRep),
]

styles, grid_cfg, use_top_legend = get_style(THEME)

# ========= 그리기 =========
fig, ax = plt.subplots(figsize=(6.4, 4.0))

for label, y in series_data:
    ax.plot(k, y, label=label, **styles[label])

# 축/범위/눈금
ax.set_xlabel("Pass@k")
ax.set_ylabel("Value(%)")
ax.set_title("", pad=6)
ax.set_xticks(k)
ax.set_ylim(18, 56)
ax.grid(True, **grid_cfg)

# 스파인 정리(더 미니멀하게)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# 범례: 상단 중앙(에디토리얼) 또는 우측 하단(기본)
if use_top_legend:
    leg = ax.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.15))
else:
    leg = ax.legend(loc="lower right", frameon=False)

# 엔드포인트 라벨(AgentRepair만 강조, 덜 난잡)
for label, y in series_data:
    if label == "AgentRepair":
        ax.annotate(f"{y[-1]:.1f}%",
                    xy=(k[-1], y[-1]),
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center", ha="left",
                    fontsize=9, color=styles[label]["color"], fontweight="bold")

plt.tight_layout()
plt.savefig(f"fig4_passk_{THEME}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"fig4_passk_{THEME}.pdf", bbox_inches="tight")
plt.show()
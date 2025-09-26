import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# ===== 폰트 설정 (IEEE 스타일: serif 계열) =====
mpl.rcParams['font.family'] = 'DejaVu Serif'

# ===== CSV 불러오기 =====
# CSV 형식: project,total_bugs,Qwen,Gemma,DeepSeek,AgentRepair
csv_path = "fig3_per_project_success_template.csv"
df = pd.read_csv(csv_path)

methods = ["Qwen", "Gemma", "DeepSeek", "AgentRepair"]

# ===== 매트릭스 계산 =====
percent_mat = []
annot_mat = []
for _, row in df.iterrows():
    row_perc = []
    row_annot = []
    for m in methods:
        cnt = int(row[m])
        tot = int(row["total_bugs"])
        pct = 100.0 * cnt / tot if tot else np.nan
        row_perc.append(pct)
        # 두 줄 버전
        row_annot.append(f"{cnt}/{tot}\n({pct:.0f}%)")
        # 한 줄 버전으로 간단히 하려면 ↓ 주석 해제
        # row_annot.append(f"{cnt}/{tot}={pct:.0f}%")
    percent_mat.append(row_perc)
    annot_mat.append(row_annot)

percent_mat = np.array(percent_mat)
annot_mat = np.array(annot_mat)

# ===== 히트맵 그리기 =====
fig, ax = plt.subplots(figsize=(6, max(4, 0.5*len(df))))

sns.heatmap(percent_mat, annot=annot_mat, fmt="", cmap="YlOrRd",
            xticklabels=methods, yticklabels=df["project"].tolist(),
            cbar_kws={"label": "Success rate (%)"}, ax=ax,
            linewidths=0.5, linecolor="white")

# heatmap 만든 뒤
cbar = ax.collections[0].colorbar
cbar.ax.set_position([0.88, 0.15, 0.02, 0.7])
# [왼쪽 x, 아래 y, 폭, 높이]  (figure 좌표 기준)

# ===== 라벨 & 타이틀 =====
ax.set_title("",
             fontsize=12, fontweight="bold", pad=12)
ax.set_xlabel("Model", fontsize=11)
ax.set_ylabel("Project", fontsize=11)

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

# heatmap 생성 후
cbar = ax.collections[0].colorbar  # 컬러바 객체 가져오기

# 컬러바 라벨과 컬러바 간격 줄이기
cbar.set_label("Success rate (%)", labelpad=1)  # 기본은 보통 10

# 컬러바 눈금과 눈금 라벨 간격 줄이기
cbar.ax.tick_params(pad=2)


plt.tight_layout()
plt.savefig("fig3_heatmap_serif.png", dpi=400, bbox_inches="tight")
plt.show()
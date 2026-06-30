
import os
import dataset
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Settings ──────────────────────────────────────────────────────────────────
DATA_FILE  = '../../Data/new-16-output-pest-1000.json'
OUT_DIR    = '../Outputfiles'
DIMS       = ['Political', 'Economic', 'Social', 'Technological']
SCORE_LABELS = [0, 1, 2, 3]

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load & filter (mirrors regressor_jesse.py / get_data) ─────────────────────
print("Loading data...")
df = dataset.get_df(DATA_FILE)

total_raw = len(df)
print(f"  Total articles loaded : {total_raw}")

if 'relevant' in df.columns:
    before = len(df)
    df = df[df['relevant'] == True].reset_index(drop=True)
    print(f"  Dropped {before - len(df)} irrelevant articles  →  {len(df)} remain")

before = len(df)
df = df.dropna(subset=DIMS).reset_index(drop=True)
if before - len(df) > 0:
    print(f"  Dropped {before - len(df)} articles with missing PEST scores  →  {len(df)} remain")

for dim in DIMS:
    df[dim] = df[dim].astype(int)

total_annotated = len(df)
print(f"\n  Final annotated dataset: {total_annotated} articles\n")

# ── Build distribution table ───────────────────────────────────────────────────
rows = []
for dim in DIMS:
    counts = df[dim].value_counts().reindex(SCORE_LABELS, fill_value=0)
    pcts   = (counts / total_annotated * 100).round(1)
    for score in SCORE_LABELS:
        rows.append({
            'Dimension': dim,
            'Score': score,
            'Count': counts[score],
            'Percentage': pcts[score]
        })

dist_df = pd.DataFrame(rows)

# ── Console output ─────────────────────────────────────────────────────────────
print("=" * 55)
print(f"{'Score distribution — annotated dataset':^55}")
print(f"{'(n = ' + str(total_annotated) + ' articles)':^55}")
print("=" * 55)

for dim in DIMS:
    sub = dist_df[dist_df['Dimension'] == dim]
    print(f"\n  {dim}")
    print(f"  {'Score':<8} {'Count':>7} {'%':>7}")
    print(f"  {'-'*24}")
    for _, row in sub.iterrows():
        print(f"  {int(row['Score']):<8} {int(row['Count']):>7} {row['Percentage']:>6.1f}%")

# ── LaTeX-ready pivot ─────────────────────────────────────────────────────────
pivot_count = dist_df.pivot(index='Dimension', columns='Score', values='Count')
pivot_pct   = dist_df.pivot(index='Dimension', columns='Score', values='Percentage')

print("\n\nLaTeX-ready table (counts / percentage):\n")
print(f"{'Dimension':<16}", end="")
for s in SCORE_LABELS:
    print(f"{'Score ' + str(s):>16}", end="")
print()
print("-" * (16 + 16 * len(SCORE_LABELS)))
for dim in DIMS:
    print(f"{dim:<16}", end="")
    for s in SCORE_LABELS:
        c = int(pivot_count.loc[dim, s])
        p = pivot_pct.loc[dim, s]
        print(f"{str(c) + ' (' + str(p) + '%)':>16}", end="")
    print()

# Save CSV
csv_path = os.path.join(OUT_DIR, 'score_distributions.csv')
dist_df.to_csv(csv_path, index=False)
print(f"\n  Distribution table saved to {csv_path}")

# ── Plot ───────────────────────────────────────────────────────────────────────
COLORS = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2']

fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=False)
fig.suptitle(
    f'PEST Score Distributions  (n = {total_annotated} annotated articles)',
    fontsize=13, fontweight='bold', y=1.02
)

for ax, dim, color in zip(axes, DIMS, COLORS):
    sub    = dist_df[dist_df['Dimension'] == dim]
    counts = sub['Count'].values
    pcts   = sub['Percentage'].values

    bars = ax.bar(SCORE_LABELS, counts, color=color, edgecolor='white', linewidth=0.8)

    # Annotate bars with percentage
    for bar, pct in zip(bars, pcts):
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + max(counts) * 0.02,
                f'{pct:.0f}%',
                ha='center', va='bottom', fontsize=9, color='#333333'
            )

    ax.set_title(dim, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('Score', fontsize=9)
    ax.set_ylabel('Article count', fontsize=9)
    ax.set_xticks(SCORE_LABELS)
    ax.set_xlim(-0.6, 3.6)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=9)

plt.tight_layout()
plot_path = os.path.join(OUT_DIR, 'score_distributions.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Plot saved to {plot_path}")

print("\nDone.")
"""
cluster_statistics.py
Reads the event-clustered output from article_to_event_level.py and prints
a full set of cluster statistics for use in your Results section.

Run from the project root after article_to_event_level.py has been run.
"""

import pandas as pd
import dataset

# ── Settings ──────────────────────────────────────────────────────────────────
OUTPUT_FILE = 'Data/09-event-clustered_new.json'

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading clustered dataset...")
df = dataset.get_df(OUTPUT_FILE)
print(f"  Loaded {len(df)} articles from {OUTPUT_FILE}\n")

# ── Basic counts ───────────────────────────────────────────────────────────────
total_articles  = len(df)
noise_articles  = int((df['event_cluster'] == -1).sum())
clustered       = total_articles - noise_articles
n_events        = df[df['event_cluster'] != -1]['event_cluster'].nunique()
noise_pct       = round(noise_articles / total_articles * 100, 1)
clustered_pct   = round(clustered / total_articles * 100, 1)

print("=" * 50)
print(f"  Total articles       : {total_articles}")
print(f"  Clustered articles   : {clustered}  ({clustered_pct}%)")
print(f"  Noise articles (-1)  : {noise_articles}  ({noise_pct}%)")
print(f"  Number of events     : {n_events}")
print("=" * 50)

# ── Per-cluster size stats ─────────────────────────────────────────────────────
cluster_sizes = (
    df[df['event_cluster'] != -1]
    .groupby('event_cluster')
    .size()
    .reset_index(name='article_count')
    .sort_values('article_count', ascending=False)
)

print(f"\n  Cluster size statistics:")
print(f"    Min articles per event  : {cluster_sizes['article_count'].min()}")
print(f"    Max articles per event  : {cluster_sizes['article_count'].max()}")
print(f"    Mean articles per event : {cluster_sizes['article_count'].mean():.1f}")
print(f"    Median                  : {cluster_sizes['article_count'].median():.1f}")
print(f"    Std dev                 : {cluster_sizes['article_count'].std():.1f}")

# ── Size distribution ──────────────────────────────────────────────────────────
print(f"\n  Article count per event (sorted largest → smallest):")
print(f"  {'Event':>8}  {'Articles':>10}")
print(f"  {'-'*22}")
for _, row in cluster_sizes.iterrows():
    print(f"  {int(row['event_cluster']):>8}  {int(row['article_count']):>10}")

# ── LaTeX-ready summary line ───────────────────────────────────────────────────
print(f"""
LaTeX summary sentence:
  The pipeline identified {n_events} distinct events from {total_articles} articles,
  with {noise_articles} ({noise_pct}\\%) articles classified as noise.
  Cluster sizes ranged from {cluster_sizes['article_count'].min()} to
  {cluster_sizes['article_count'].max()} articles
  (mean {cluster_sizes['article_count'].mean():.1f}, 
  median {cluster_sizes['article_count'].median():.0f}).
""")
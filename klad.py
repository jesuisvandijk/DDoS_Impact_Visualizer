"""
diagnose_clustering.py

Quick diagnostic for the event-clustering pipeline.

Tests the hypothesis that entity injection in embed_articles() is
artificially shrinking the distance between articles that get wrongly
merged into the same event cluster.

For each suspect group of article indices, prints:
  - cosine distance using CONTENT TEXT ONLY
  - cosine distance using CONTENT + INJECTED ENTITIES (current pipeline)
  - the actual entity strings being appended

If "with entities" distance is noticeably smaller than "content only" for
a group you don't believe belongs together, that's the signature of the bug.
Compare both against the random baseline at the bottom for context on what
a "typical" distance between unrelated articles looks like.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer
from article_to_event_level import EMBEDDER

# ── Settings ──────────────────────────────────────────────────────────────

# Use the clustered output file -- it already has Content + the ent_* columns.
# Point this at whatever file currently holds your event_cluster results.
DATA_FILE = 'Data/09-event-clustered_new.json'
content_col = 'Content'

# Edit these to whichever article indices you want to inspect.
# Indices should match the row position in DATA_FILE (same indexing used
# throughout run_pipeline -- News-filtered, reset_index(drop=True)).
SUSPECT_GROUPS = {
    'event_18_suspected_bad': [430, 443, 439],
    'event_31_suspected_bad': [157, 233, 259],
}

ENTITY_COLS = ['ent_orgs', 'ent_gpe', 'ent_dates', 'ent_nums',
               'ent_malware', 'ent_indicator', 'ent_system']

RANDOM_BASELINE_PAIRS = 30  # for context: typical distance between random articles

# ── Helpers ───────────────────────────────────────────────────────────────

def entity_string(row):
    parts = []
    for col in ENTITY_COLS:
        parts += row.get(col, []) or []
    return ' '.join(parts)


def print_pairwise(label, dist_matrix, names):
    print(f"\n  {label}")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            print(f"    {names[i]:>6} <-> {names[j]:<6}: {dist_matrix[i, j]:.3f}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {DATA_FILE} ...")
    df = pd.read_json(DATA_FILE)

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDER)

    for group_name, indices in SUSPECT_GROUPS.items():
        print(f"\n{'=' * 60}\nGROUP: {group_name}  indices={indices}\n{'=' * 60}")

        rows = df.loc[indices]
        names = [f"#{i}" for i in indices]

        content_texts = rows[content_col].fillna('').tolist()
        entity_strs = rows.apply(entity_string, axis=1).tolist()
        combined_texts = [
            (c + ' ' + e).strip() for c, e in zip(content_texts, entity_strs)
        ]

        for idx, e in zip(indices, entity_strs):
            print(f"  #{idx} entities injected: {e[:200]}")

        dist_content_only = cosine_distances(model.encode(content_texts))
        dist_with_entities = cosine_distances(model.encode(combined_texts))

        print_pairwise("Distance — CONTENT ONLY", dist_content_only, names)
        print_pairwise("Distance — CONTENT + ENTITIES (current pipeline)", dist_with_entities, names)

    print(f"\n{'=' * 60}\nRANDOM BASELINE ({RANDOM_BASELINE_PAIRS} pairs)\n{'=' * 60}")
    sample = df[content_col].fillna('').sample(
        n=min(RANDOM_BASELINE_PAIRS * 2, len(df)), random_state=0
    ).tolist()
    rand_dist = cosine_distances(model.encode(sample))
    upper = rand_dist[np.triu_indices_from(rand_dist, k=1)]
    print(f"  Mean: {upper.mean():.3f}   Min: {upper.min():.3f}   Max: {upper.max():.3f}")


if __name__ == '__main__':
    main()
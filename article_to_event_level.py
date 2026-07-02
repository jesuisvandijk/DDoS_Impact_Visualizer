import dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import subprocess
import sys
import re

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline as hf_pipeline
from tqdm import tqdm

from ollama import Client
import hdbscan
import spacy
import umap

nlp = spacy.load('en_core_web_sm')
EMBEDDER = 'paraphrase-multilingual-MiniLM-L12-v2'
# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

DATA_FILE       = 'Data/new-16-output-pest-1000.json'
OUTPUT_FILE     = 'Data/09-event-clustered_new.json'

# ─────────────────────────────────────────────────────────────────────────────
# Cached files - clear/delete when using a new dataset
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDINGS_FILE = 'Cache/embeddings_new.npy'
ENTITIES_FILE = 'Cache/entities_cache_new.json'
EVENT_SUMMARY_FILE = 'Cache/event_summary.json'

# ─────────────────────────────────────────────────────────────────────────────
# Settings — adjust for dataset
# ─────────────────────────────────────────────────────────────────────────────

HDBSCAN_MIN_CLUSTER_SIZE = 4  # minimum articles to form an event
                                # increase if too many tiny clusters
                                # decrease if too much noise (-1)
HDBSCAN_MIN_SAMPLES      = 3 # higher = more conservative, more noise

MAX_CLUSTER_SIZE = 50   # clusters larger than this get sub-clustered automatically

SUB_MCS = 4             # min_cluster_size for sub-clustering
SUB_MS  = 2             # min_samples for sub-clustering

UMAP_N_NEIGHBORS = 15    # 5-50, lower = more local structure
UMAP_MIN_DIST    = 0.15   # 0.0-0.5, lower = tighter clusters

# ─────────────────────────────────────────────────────────────────────────────
# Toggle descriptions, dashboard and graphs
# ─────────────────────────────────────────────────────────────────────────────

GENERATE_EVENT_DESCRIPTION = True #Toggle to generate the description for events, takes a very long time
LAUNCH_DASH = True #Toggle to automatically launch dashboard
SHOW_GRAPHS = False #Show all files in screen

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Named Entity Recognition
# ─────────────────────────────────────────────────────────────────────────────

_tokenizer = AutoTokenizer.from_pretrained('AI4Sec/cyner-xlm-roberta-base')
_model = AutoModelForTokenClassification.from_pretrained('AI4Sec/cyner-xlm-roberta-base')
_ner_pipeline = hf_pipeline('ner', model=_model, tokenizer=_tokenizer, aggregation_strategy='simple')

def extract_entities(text_content):
    entities = {}

    predictions = _ner_pipeline(text_content[:512])
    for ent in predictions:
        label = ent['entity_group']
        entities.setdefault(label, []).append(ent['word'])

    doc = nlp(text_content)
    for ent in doc.ents:
        if ent.label_ in ('GPE', 'DATE', 'CARDINAL', 'LOC', 'PERSON'):
            entities.setdefault(ent.label_, []).append(ent.text)

    return entities

def add_entities_to_df(df, content_col='Content'):
    df = df.copy()

    cache_file = f'Cache/entities_cache_{len(df)}.json'

    if os.path.exists(cache_file):
        print("  Loading cached entities...")
        with open(cache_file) as f:
            entities = json.load(f)
    else:
        print("  Extracting entities (cyNER + spaCy)...")
        entities = df[content_col].apply(extract_entities).tolist()
        with open(cache_file, 'w') as f:
            json.dump(entities, f)
        print(f"  Entities cached to {cache_file}")

    df[content_col] = df[content_col].str.replace(r'_x[0-9A-Fa-f]{4}_', ' ', regex=True)
    df['entities'] = entities
    df['ent_orgs']      = df['entities'].apply(lambda e: e.get('Organization', []))
    df['ent_gpe']       = df['entities'].apply(lambda e: e.get('GPE', []))
    df['ent_dates']     = df['entities'].apply(lambda e: e.get('DATE', []))
    df['ent_nums']      = df['entities'].apply(lambda e: e.get('CARDINAL', []))
    df['ent_malware']   = df['entities'].apply(lambda e: e.get('Malware', []))
    df['ent_indicator'] = df['entities'].apply(lambda e: e.get('Indicator', []))
    df['ent_system']    = df['entities'].apply(lambda e: e.get('System', []))
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Sentence Embeddings
# ─────────────────────────────────────────────────────────────────────────────


def embed_articles(df, content_col='Content'):
    model = SentenceTransformer(EMBEDDER)

    def build_input(row):
        base = row[content_col] or ''
        entities = []
        for col in ["ent_orgs", "ent_malware", "ent_indicator", "ent_system", "ent_gpe", "ent_dates"]:  # dropped ent_gpe, ent_dates
            entities += row.get(col, [])
        entities = list(dict.fromkeys(entities))
        if entities:
            return base + ' ' + ' '.join(entities)
        return base

    texts = df.apply(build_input, axis=1).tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Dimensionality Reduction (UMAP)
# ─────────────────────────────────────────────────────────────────────────────

def reduce_dimensions(embeddings, n_components, random_state=42):
    """
    Reduces high-dimensional embeddings to a lower number of dimensions.
    """
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        random_state=random_state
    )
    return reducer.fit_transform(embeddings)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Clustering (HDBSCAN)
# ─────────────────────────────────────────────────────────────────────────────

def cluster_articles(embeddings):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric='cosine',
        algorithm='generic',
        cluster_selection_method='eom'
    )
    return clusterer.fit_predict(embeddings.astype('float64'))

# ─────────────────────────────────────────────────────────────────────────────
# Stage 4b — Automatic sub-clustering of oversized events
# ─────────────────────────────────────────────────────────────────────────────

def split_large_clusters(df, embeddings):
    iteration = 0
    while True:
        iteration += 1
        cluster_sizes = df[df['event_cluster'] != -1]['event_cluster'].value_counts()
        large_clusters = cluster_sizes[cluster_sizes > MAX_CLUSTER_SIZE].index.tolist()

        if not large_clusters:
            print(f"  No clusters exceed size {MAX_CLUSTER_SIZE} — done after {iteration - 1} iteration(s).")
            break

        print(f"\n  Iteration {iteration}: {len(large_clusters)} oversized cluster(s): {large_clusters}")

        for cluster_id in large_clusters:
            mask    = df['event_cluster'] == cluster_id
            indices = np.where(mask)[0]
            sub_embeddings = embeddings[indices]

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=SUB_MCS,
                min_samples=SUB_MS,
                metric='cosine',
                algorithm='generic',
                cluster_selection_method='eom'
            )
            sub_labels = clusterer.fit_predict(sub_embeddings.astype('float64'))

            n_sub   = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
            n_noise = int(sum(sub_labels == -1))

            if n_sub <= 1:
                # Sub-clustering found no meaningful split — leave this cluster alone
                print(f"    Cluster {cluster_id} ({len(indices)} articles): "
                      f"could not be split further, leaving as-is.")
                df.loc[mask, 'event_cluster'] = -(cluster_id + 2)
                continue

            max_existing = df['event_cluster'].max()
            new_labels = np.where(
                sub_labels == -1,
                -1,
                sub_labels + max_existing + 1
            )
            df.loc[mask, 'event_cluster'] = new_labels
            print(f"    Cluster {cluster_id} ({len(indices)} articles) → "
                  f"{n_sub} sub-events, {n_noise} pushed to noise.")

    exempt_mask = df['event_cluster'] < -1
    if exempt_mask.any():
        exempt_ids = df.loc[exempt_mask, 'event_cluster'].unique()
        max_id = df[df['event_cluster'] >= 0]['event_cluster'].max()
        for old_id in exempt_ids:
            max_id += 1
            df.loc[df['event_cluster'] == old_id, 'event_cluster'] = max_id
            print(f"  Restored exempt cluster {old_id} → {max_id}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_clusters(reduced_2d, labels, title='Article Event Clusters'):
    fig, ax = plt.subplots(figsize=(14, 9))
    colors = plt.cm.tab20.colors

    unique_labels = sorted(set(labels))
    for i, label in enumerate(unique_labels):
        mask  = labels == label
        color = 'lightgrey'    if label == -1 else colors[i % len(colors)]
        name  = 'Noise (-1)'   if label == -1 else f'Event {label}'
        ax.scatter(
            reduced_2d[mask, 0],
            reduced_2d[mask, 1],
            c=[color],
            label=name,
            alpha=0.6,
            s=12
        )

    ax.legend(loc='upper right', markerscale=2, fontsize=7, ncol=2)
    ax.set_title(title)
    ax.set_xlabel('UMAP dimension 1')
    ax.set_ylabel('UMAP dimension 2')
    plt.tight_layout()
    plt.savefig('Dashboard/Outputfiles/event_clusters.png', format='png')
    if SHOW_GRAPHS:
        plt.show()
    print("  Cluster plot saved to event_clusters.png")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 — PESTLE aggregation per event
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_pestle_per_event(df):
    pestle_cols = [
        'Political', 'Economic', 'Social',
        'Technological'
    ]
    num_col = len(pestle_cols)
    available = [c for c in pestle_cols if c in df.columns]

    if not available:
        print("  No PESTLE columns found — skipping aggregation.")
        print("  Run ollama_annotate.py first, then re-run this pipeline.")
        return None

    df_events = df[df['event_cluster'] != -1]
    agg_dict  = {'Content': 'count'}
    agg_dict.update({col: 'mean' for col in available})

    event_summary = (
        df_events
        .groupby('event_cluster')
        .agg(agg_dict)
        .rename(columns={'Content': 'article_count'})
        .reset_index()
    )
    angles = [n / num_col * 2 * np.pi for n in range(num_col)]
    angles += angles[:1]

    for _, row in event_summary.iterrows():
        cluster_id = int(row['event_cluster'])
        values = [row[f'{dim}'] for dim in pestle_cols]
        values += values[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        ax.plot(angles, values, linewidth=2, color='steelblue')
        ax.fill(angles, values, alpha=0.25, color='steelblue')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(pestle_cols, fontsize=12)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['0', '1', '2', '3'], fontsize=8)
        ax.set_ylim(0, 3)

        ax.set_title(f'Event {cluster_id} — PESTLE impact\n'
                     f'({int(row["article_count"])} articles)',
                     pad=20)

        plt.tight_layout()
        plt.savefig(f'Dashboard/Outputfiles/radar_event_{cluster_id}.png', format='png')
        if SHOW_GRAPHS:
            plt.show()
            print(f"  Radar chart saved: radar_event_{cluster_id}.png")
    return event_summary

# ─────────────────────────────────────────────────────────────────────────────
# Stage 7 - Optional: Generate description per event
# ─────────────────────────────────────────────────────────────────────────────

def generate_description(event_summary, df):
    client = Client(host="http://localhost:11434")
    descriptions = []

    system_prompt = """You are a cybersecurity analyst who writes concise, factual incident summaries.

STRICT RULES:
- Respond ONLY in English, even if the source articles are in another language. Never output Dutch, German, French, or any other language.
- Output exactly 3 sentences and nothing else.
- Do not include any preamble, introduction, acknowledgement, or meta-commentary (e.g. no "Here is a summary", "Based on the articles", "Sure,", "Okay,").
- Do not repeat these instructions or mention that you are an AI.
- Start the first sentence immediately with the subject of the event (e.g. "A DDoS attack targeted...").
- Sentence 1: what happened. Sentence 2: who was affected. Sentence 3: key technical details.
- If the articles are unclear or conflicting, summarize the most consistent version of events rather than commenting on the inconsistency."""

    for _, row in tqdm(event_summary.iterrows(), total=len(event_summary), desc="Generating descriptions"):
        cluster_id = int(row["event_cluster"])
        articles = df[df["event_cluster"] == cluster_id]
        article_texts = articles["Content"].dropna().tolist()
        combined = "\n\n---\n\n".join(article_texts[:10])

        prompt = f"""Below are {len(article_texts)} news articles (possibly in different languages) describing the same DDoS event.

Articles:
{combined}

Write the 3-sentence English summary now, following the rules exactly. Begin your response with the first word of sentence 1 — no other text before it."""

        response = client.chat(
            model="llama3.2:1b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            options={"num_ctx": 8192, "num_predict": 200, "temperature": 0.3}
        )

        text = response.message.content.strip()
        text = clean_description(text)
        descriptions.append(text)
        print(f"Description for event {cluster_id} generated.")

    event_summary["description"] = descriptions
    return event_summary


def clean_description(text: str) -> str:
    preamble_patterns = [
        r"^(sure|okay|ok|certainly|here is|here's|based on|summary:|as requested)[^.]*?:\s*",
        r"^(sure|okay|ok|certainly)[,.\s]+",
    ]
    cleaned = text
    for pattern in preamble_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    return cleaned

# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline():
    # ── Step 1: Load ─────────────────────────────────────────────────────────
    print("\nStep 1: Loading data...")
    df = dataset.get_df(DATA_FILE)
    df = df[df['Alert Type'] == 'News'].reset_index(drop=True)
    if 'relevant' in df.columns:
        before = len(df)
        df = df[df['relevant'] == True].reset_index(drop=True)
        print(f"  Dropped {before - len(df)} non-cyber articles before clustering.")
    print(f"  {len(df)} news articles loaded from {DATA_FILE}.")

    # ── Step 2: NER ───────────────────────────────────────────────────────────
    print("\nStep 2: Extracting named entities...")
    df = add_entities_to_df(df)
    print(f"  Done")

    # ── Step 3: Embed articles ────────────────────────────────────────────────
    print("\nStep 3: Embedding articles...")
    embeddings = None
    if os.path.exists(EMBEDDINGS_FILE):
        cached = np.load(EMBEDDINGS_FILE)
        if cached.shape[0] == len(df):
            embeddings = cached
            print(f"  Loaded cached embeddings from {EMBEDDINGS_FILE}.")
        else:
            print(f"  Cached embeddings have {cached.shape[0]} rows but current dataset has {len(df)} — recomputing.")

    if embeddings is None:
        embeddings = embed_articles(df)
        np.save(EMBEDDINGS_FILE, embeddings)
        print(f"  Embeddings saved to {EMBEDDINGS_FILE} for future runs.")

    print(f"  Embedding shape: {embeddings.shape}")

    # ── Step 4: Cluster ───────────────────────────────────────────────────────
    print("\nStep 4: Clustering articles into events (HDBSCAN)...")
    labels = cluster_articles(embeddings)
    df['event_cluster'] = labels

    # ── Step 4b: Split oversized clusters ────────────────────────────────────
    print("\nStep 4b: Splitting oversized clusters...")
    df = split_large_clusters(df, embeddings)

    # Recompute labels and stats after splitting
    labels = df['event_cluster'].values
    n_events = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(sum(labels == -1))
    print(f"\n  Final: {n_events} events, "
          f"{n_noise} noise articles ({round(n_noise / len(df) * 100, 1)}%)")

    # ── Step 5: Visualise ─────────────────────────────────────────────────────
    print("\nStep 5: Visualising clusters...")
    reduced_2d = reduce_dimensions(embeddings, n_components=2)  # 2D for plot only
    plot_clusters(reduced_2d, labels)

    # ── Step 6: PESTLE aggregation ────────────────────────────────────────────
    print("\nStep 7: Aggregating PESTLE scores per event...")
    event_summary = aggregate_pestle_per_event(df)

    if GENERATE_EVENT_DESCRIPTION:
        print("\nGenerating event summary... ")
        event_summary = generate_description(event_summary, df)
    else:
        if event_summary is not None and os.path.exists(EVENT_SUMMARY_FILE):
            try:
                old_summary = pd.read_json(EVENT_SUMMARY_FILE)
                if 'description' in old_summary.columns:
                    event_summary = event_summary.merge(
                        old_summary[['event_cluster', 'description']],
                        on='event_cluster', how='left'
                    )
                    print("  Reused existing descriptions from previous run.")
            except Exception as e:
                print(f"  Could not load previous descriptions: {e}")

    if event_summary is not None:
        event_summary.to_json(EVENT_SUMMARY_FILE, orient='columns')
        print(f"  Event summary (with descriptions) saved to {EVENT_SUMMARY_FILE}")
        print(event_summary.to_string(index=False))

    # ── Step 7: open streamlit  ────────────────────────────────────────────────
    if LAUNCH_DASH:
        print("\nLaunching dashboard...")
        subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "streamlit_app.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    df_save = df.drop(columns=['entities'], errors='ignore')
    df_save.to_json(OUTPUT_FILE, orient='columns')
    print(f"\nDone! Results saved to {OUTPUT_FILE}")

    return df, event_summary


if __name__ == '__main__':
    df, event_summary = run_pipeline()
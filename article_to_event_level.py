import dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import subprocess
import sys

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline as hf_pipeline
from tqdm import tqdm
from sklearn.metrics import pairwise_distances


from ollama import Client
import hdbscan
import spacy
import umap

nlp = spacy.load('en_core_web_sm')
EMBEDDER = 'paraphrase-multilingual-MiniLM-L12-v2'
# ─────────────────────────────────────────────────────────────────────────────
# Settings — adjust for dataset
# ─────────────────────────────────────────────────────────────────────────────

DATA_FILE       = 'Data/new-15-output-pest-1000.json'  # annotated training dataset
EMBEDDINGS_FILE = 'Cache/embeddings_new.npy'  # cached embeddings
OUTPUT_FILE     = 'Data/09-event-clustered_new.json'
ENTITIES_FILE = 'Cache/entities_cache_new.json'

# DBSCAN parameters — tune if clusters look wrong
# eps (base = 0.5):   neighbourhood radius — lower = more clusters, higher = fewer
# min_samples (base = 5): minimum articles to form a cluster
#DBSCAN_EPS         = 0.5
#DBSCAN_MIN_SAMPLES = 5

HDBSCAN_MIN_CLUSTER_SIZE = 3   # minimum articles to form an event
                                # increase if too many tiny clusters
                                # decrease if too much noise (-1)
HDBSCAN_MIN_SAMPLES      = 3   # higher = more conservative, more noise

# UMAP parameters
UMAP_N_NEIGHBORS = 8    # 5-50, lower = more local structure
UMAP_MIN_DIST    = 0.05   # 0.0-0.5, lower = tighter clusters

#Toggle to generate the description for events, takes a very long time
GENERATE_EVENT_DESCRIPTION = False

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

def build_combined_distance(reduced_embeddings, dates, date_weight=0.4, max_days=14):
    sem_dist = pairwise_distances(reduced_embeddings, metric='cosine')
    days = np.array([(d1 - d2).days for d1 in dates for d2 in dates]).reshape(len(dates), len(dates))
    date_dist = np.clip(np.abs(days) / max_days, 0, 1)  # normalized 0-1, capped at max_days
    combined = (1 - date_weight) * sem_dist + date_weight * date_dist
    return combined

def cluster_articles(combined_distance):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric='precomputed',
        cluster_selection_method='leaf'
    )
    return clusterer.fit_predict(combined_distance.astype('float64'))

#def cluster_articles(reduced_embeddings):
    """
    Clusters articles into events using HDBSCAN.
    Tune HDBSCAN_MIN_CLUSTER_SIZE at the top of this file:
      - Too many tiny clusters  → increase HDBSCAN_MIN_CLUSTER_SIZE
      - Everything is noise (-1) → decrease HDBSCAN_MIN_CLUSTER_SIZE
      - Too few broad clusters  → decrease HDBSCAN_MIN_SAMPLES
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric='cosine',
        algorithm = 'generic'
    )
    labels = clusterer.fit_predict(reduced_embeddings.astype('float64'))
    return labels

# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_clusters(reduced_2d, labels, title='Article Event Clusters'):
    """
    Scatter plot of articles in 2D UMAP space, coloured by event cluster.
    Noise articles (label -1) are shown in light grey.
    Saved as event_clusters.png.
    """
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
#    plt.show()
    print("  Cluster plot saved to event_clusters.png")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 — PESTLE aggregation per event
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_pestle_per_event(df):
    """
    Groups articles by event cluster and computes mean PESTLE scores per event.
    Only runs if PESTLE annotation columns are present in the dataframe.
    Automatically creates a radar chart of the aggregated values
    Returns a summary dataframe with one row per event.
    """
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
    # Compute the angle for each dimension on the circle
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
#        plt.show()
#        print(f"  Radar chart saved: radar_event_{cluster_id}.png")
    return event_summary

# ─────────────────────────────────────────────────────────────────────────────
# Stage 7 - Optional: Generate description per event
# ─────────────────────────────────────────────────────────────────────────────

def generate_description(event_summary, df):
    client = Client(host="http://localhost:11434")
    descriptions = []

    for _, row in tqdm(event_summary.iterrows(), total=len(event_summary), desc="Generating descriptions"):
        cluster_id = int(row["event_cluster"])
        articles = df[df["event_cluster"] == cluster_id]
        article_texts = articles["Content"].dropna().tolist()
        combined = "\n\n---\n\n".join(article_texts[:10])

        prompt = f"""You are given {len(article_texts)} news articles that belong to the 
        same DDoS event cluster. 
        Summarize this DDoS event in English in 3 sentences: 
        what happened, who was affected, key technical details. 
        Start directly with the description, no introduction, presenting or affirmation.

        Articles:
        {combined}"""

        response = client.chat(
            model="llama3.2:1b",
            messages=[{"role": "system", "content": "You are a cybersecurity analyst. Always respond in English, regardless of what language the source articles are written in."},
                      {"role": "user", "content": prompt}],
            options={"num_ctx": 8192, "num_predict": 200}
        )
        descriptions.append(response.message.content)
        print(f"Description for event {cluster_id} generated.")
    event_summary["description"] = descriptions
    return event_summary

# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline():
    # ── Step 1: Load ─────────────────────────────────────────────────────────
    print("\nStep 1: Loading data...")
    df = dataset.get_df(DATA_FILE)
    df = df.drop_duplicates(subset='Content').reset_index(drop=True)
    df = df[df['Alert Type'] == 'News'].reset_index(drop=True)
    print(f"  {len(df)} news articles loaded from {DATA_FILE}.")


    # ── Step 2: NER ───────────────────────────────────────────────────────────
    print("\nStep 2: Extracting named entities...")
    df = add_entities_to_df(df)
    print(f"  Done")

    # ── Step 3: Embed ─────────────────────────────────────────────────────────
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


    # ── Step 4: Reduce dimensions ─────────────────────────────────────────────
    print("\nStep 4: Reducing dimensions with UMAP...")
    print("  Computing 10D reduction for clustering...")
    reduced_10d = reduce_dimensions(embeddings, n_components=10)
    print("  Computing 2D reduction for visualisation...")
    reduced_2d  = reduce_dimensions(embeddings, n_components=2)

    # ── Step 5: Cluster ───────────────────────────────────────────────────────
    print("\nStep 5: Clustering articles into events (HDBSCAN)...")
    dates = pd.to_datetime(df['Date'], errors='coerce')  # adjust column name if yours differs
    combined_distance = build_combined_distance(reduced_10d, dates)
    labels = cluster_articles(combined_distance)

    df['event_cluster'] = labels
    n_events = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise  = int(sum(labels == -1))
    print(f"  Events found:   {n_events}")
    print(f"  Noise articles: {n_noise} ({round(n_noise / len(df) * 100, 1)}%)")
    print(f"  Tip: adjust HDBSCAN_EPS at the top of this file if results look wrong.")

    # ── Step 6: Visualise ─────────────────────────────────────────────────────
    print("\nStep 6: Visualising clusters...")
    plot_clusters(reduced_2d, labels)

    # ── Step 7: PESTLE aggregation ────────────────────────────────────────────
    print("\nStep 7: Aggregating PESTLE scores per event...")
    event_summary = aggregate_pestle_per_event(df)

    if GENERATE_EVENT_DESCRIPTION:
        print("\nGenerating event summary... ")
        event_summary = generate_description(event_summary, df)

    from streamlit_app import EVENT_SUMMARY_FILE

    if event_summary is not None:
        event_summary.to_json(EVENT_SUMMARY_FILE, orient='columns')
        print(f"  Event summary (with descriptions) saved to {EVENT_SUMMARY_FILE}")
        print(event_summary.to_string(index=False))

    # ── Step 8: open streamlit  ────────────────────────────────────────────────
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
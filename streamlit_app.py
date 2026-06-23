import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from ollama import Client
import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from dataset import get_df
from article_to_event_level import aggregate_pestle_per_event
from article_to_event_level import OUTPUT_FILE, EVENT_SUMMARY_FILE

LOAD_DESCRIPTIONS = True

# ── Config ────────────────────────────────────────────────────────
DATA_FILE = OUTPUT_FILE
PESTLE_DIMS = ['Political', 'Economic', 'Social', 'Technological']


# ── Load data (cached so it only runs once) ───────────────────────
@st.cache_data
def load_data():
    df = get_df(DATA_FILE)
    if os.path.exists(EVENT_SUMMARY_FILE):
        event_summary = pd.read_json(EVENT_SUMMARY_FILE)
    else:
        st.warning("No saved event summary with descriptions found — falling back to live aggregation (no descriptions).")
        event_summary = aggregate_pestle_per_event(df)
    return df, event_summary

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_data
def embed_descriptions(descriptions):
    model = load_embedding_model()
    return model.encode(descriptions.tolist())

load_data.clear()
embed_descriptions.clear()


df, event_summary = load_data()

events_overview = []

# ── Session state init ────────────────────────────────────────────
if 'selected_event' not in st.session_state:
    st.session_state.selected_event = None

# ── Page routing ──────────────────────────────────────────────────
if st.session_state.selected_event is None:

    # OVERVIEW PAGE
    st.title('DDoS impact visualizer')
    st.write("""Welcome to the DDoS impact visualizer. This application displays the multidimensional impact of DDoS events, as gathered from news articles about these events.
    The information is annotated using an LLM agent, and the relative numbers are represented in the following rubric:""")

    st.markdown("""
    **SCALE:**  
    0 = No meaningful content relating to this dimension  
    1 = Minor or indirect mention (topic is peripheral)  
    2 = Moderate coverage (topic is discussed but not the main focus)  
    3 = Primary focus (topic is central to the article)
    """)

    political, economical, social, technological = st.columns(4)

    if political.button("Political"):
        st.markdown("""[Political]  
            Damage: national stability, geopolitical tension, political strategy regarding cyber attacks.  
Indicators: cyber attacks as tools in geopolitical rivalry and hybrid war; influence on national
security agendas and international norms; political reaction.  
  0 – Article covers a technical vulnerability with no political actors or reactions mentioned.  
  1 – A government agency is named as affected, but political implications are not discussed.  
  2 – Article discusses a nation-state attribution or calls for a policy response.  
  3 – Article centers on geopolitical conflict, diplomatic fallout, or national security strategy
      driven by a cyber event.""")
    if economical.button("Economic"):
        st.markdown("""[ECONOMIC]  
Damage: reputational damage, revenue lost, recovery costs, fines, stock price lowers.  
Indicators: macro loss modelling of extreme cyber events; firm level and sectoral losses;
critical infrastructure disruptions (e.g., ports, energy).  
  0 – No financial figures, losses, or economic actors mentioned.  
  1 – A company is named as victim but financial impact is not quantified or discussed.  
  2 – Article reports estimated losses, fines, or disruption to a specific firm or sector.  
  3 – Article focuses on large-scale financial damage, systemic economic risk, or critical
      infrastructure with major economic consequences.""")
    if social.button("Social"):
        st.markdown("""[SOCIAL]  
Damage: societal cohesion, public distress.  
Indicators: psychological distress, loss of trust, weakened organisational cohesion, societal
disruption from critical infrastructure attacks.  
  0 – No mention of public impact, trust, or civilian disruption.  
  1 – Brief mention of public concern or inconvenience without elaboration.  
  2 – Article discusses erosion of trust or distress in a specific community or organisation.  
  3 – Article centers on widespread societal disruption, mass loss of public trust, or
      psychological impact at scale.""")
    if technological.button("Technological"):
        st.markdown("""[TECHNOLOGICAL]  
Damage: physical/digital harm, infrastructure disruption.  
Indicators: availability, integrity, and confidentiality impacts; critical infrastructure
inoperability; socio-technical sophistication.
  0 – No technical detail; article is purely political or economic commentary with no system impact.  
  1 – Attack type is named (e.g. "ransomware") but technical details are absent.  
  2 – Article describes attack vectors, affected systems, or partial infrastructure disruption.  
  3 – Article provides in-depth technical analysis, covers novel TTPs, or reports full
      infrastructure inoperability.""")

else:

    # EVENT DETAIL PAGE
    if st.button("← Back"):
        st.session_state.selected_event = None
        st.rerun()

    event_id = st.session_state.selected_event
    st.title(f"Information about event {event_id}")

    articles = df[df["event_cluster"] == event_id]

    col1, col2 = st.columns(2)
    with col1:
        # Radar chart
        st.image(f"Dashboard/Outputfiles/radar_event_{event_id}.png")

    with col2:
        # Event description
        with st.spinner("Generating..."):
            st.write("test")
            if LOAD_DESCRIPTIONS:
                description = event_summary[event_summary["event_cluster"] == event_id]["description"].iloc[0]
                st.write(description)


    st.markdown("""
    **SCALE:**  
    0 = No meaningful content relating to this dimension  
    1 = Minor or indirect mention (topic is peripheral)  
    2 = Moderate coverage (topic is discussed but not the main focus)  
    3 = Primary focus (topic is central to the article)
    """)

    # Article list
    articles = df[df["event_cluster"] == event_id]

    available_cols = [c for c in ["Content_Title", "Content", "Date", "Link",
                                  "Political", "Economic", "Social", "Technological",
                                  "ent_orgs", "ent_malware", "ent_indicator", "ent_system"]
                      if c in articles.columns]

    selected_cols = st.multiselect(
        "Columns to display",
        options=available_cols,
        default=["Content", "Date", "Link"]
    )

    if selected_cols:
        st.dataframe(articles[selected_cols])
    else:
        st.info("Select at least one column.")

with st.sidebar:
    query = st.text_input("Search events by keyword", placeholder="e.g. ransomware, DDoS, Russia...")

    if query:
        model = load_embedding_model()
        desc_embeddings = embed_descriptions(event_summary["description"])

        query_embedding = model.encode([query])
        similarities = cosine_similarity(query_embedding, desc_embeddings)[0]

        threshold = 0.2
        event_summary["similarity"] = similarities
        filtered_summary = event_summary[event_summary["similarity"] > threshold].sort_values("similarity",
                                                                                              ascending=False)

        if filtered_summary.empty:
            st.info(f"No events found for '{query}'")
    else:
        filtered_summary = event_summary

    for _, row in filtered_summary.iterrows():
        event_id = int(row["event_cluster"])
        st.write(f"Event {event_id} — {int(row['article_count'])} articles")

        if LOAD_DESCRIPTIONS:
            st.caption(row["description"][:150] + "...")

        if st.button(f"View event {event_id} →", key=f"btn_{event_id}"):
            st.session_state.selected_event = event_id
            st.rerun()

    events_overview = pd.DataFrame(events_overview)
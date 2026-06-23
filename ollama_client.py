import json
import re
from ollama import Client

# PESTLE dimensions (excluding Legal and Environmental)
DIMENSIONS = ["Political", "Economic", "Social", "Technological"]

SYSTEM_PROMPT = """You are an expert analyst specializing in cybersecurity impact assessment.
You will read a news article (title + body) and perform two steps.

---
STEP 1 — RELEVANCE CHECK
Determine whether the article is substantively about a cybersecurity incident, cyber attack,
DDoS event, data breach, or directly related cyber threat activity.

Articles that are NOT relevant include (non-exhaustive): general IT/business news with no
attack or threat content, product announcements, unrelated political/economic news, sports,
entertainment, or any article where cybersecurity is not a real subject of the text.

---
STEP 2 — SCORING SCALE (only if relevant)
0 = No meaningful content relating to this dimension
1 = Minor or indirect mention (topic is peripheral)
2 = Moderate coverage (topic is discussed but not the main focus)
3 = Primary focus (topic is central to the article)

---
DIMENSIONS:

[POLITICAL]
Damage: national stability, geopolitical tension, political strategy regarding cyber attacks.
Indicators: cyber attacks as tools in geopolitical rivalry and hybrid war; influence on national
security agendas and international norms; political reaction.
  0 – Article covers a technical vulnerability with no political actors or reactions mentioned.
  1 – A government agency is named as affected, but political implications are not discussed.
  2 – Article discusses a nation-state attribution or calls for a policy response.
  3 – Article centers on geopolitical conflict, diplomatic fallout, or national security strategy
      driven by a cyber event.

[ECONOMIC]
Damage: reputational damage, revenue lost, recovery costs, fines, stock price lowers.
Indicators: macro loss modelling of extreme cyber events; firm level and sectoral losses;
critical infrastructure disruptions (e.g., ports, energy).
  0 – No financial figures, losses, or economic actors mentioned.
  1 – A company is named as victim but financial impact is not quantified or discussed.
  2 – Article reports estimated losses, fines, or disruption to a specific firm or sector.
  3 – Article focuses on large-scale financial damage, systemic economic risk, or critical
      infrastructure with major economic consequences.

[SOCIAL]
Damage: societal cohesion, public distress.
Indicators: psychological distress, loss of trust, weakened organisational cohesion, societal
disruption from critical infrastructure attacks.
  0 – No mention of public impact, trust, or civilian disruption.
  1 – Brief mention of public concern or inconvenience without elaboration.
  2 – Article discusses erosion of trust or distress in a specific community or organisation.
  3 – Article centers on widespread societal disruption, mass loss of public trust, or
      psychological impact at scale.

[TECHNOLOGICAL]
Damage: physical/digital harm, infrastructure disruption.
Indicators: availability, integrity, and confidentiality impacts; critical infrastructure
inoperability; socio-technical sophistication.
  0 – No technical detail; article is purely political or economic commentary with no system impact.
  1 – Attack type is named (e.g. "ransomware") but technical details are absent.
  2 – Article describes attack vectors, affected systems, or partial infrastructure disruption.
  3 – Article provides in-depth technical analysis, covers novel TTPs, or reports full
      infrastructure inoperability.

INSTRUCTIONS:
- Base your decision and scores on the article content only. Do not infer beyond what is written.
- When uncertain whether an article is relevant, prefer marking it NOT relevant (false negative)
  over scoring an off-topic article (false positive). Precision matters more than recall here.
- IMPORTANT: null is ONLY used when "relevant" is false, and ONLY for all four dimensions at once.
- If "relevant" is true, you MUST give every single dimension an integer score (0, 1, 2, or 3).
  A dimension having NO content is a score of 0 — it is NEVER null. Do not output null or None for a
  dimension just because that dimension isn't discussed in the article; output 0 instead.
- Weight sustained, substantive coverage more than passing mentions.
- If the title contradicts the body, follow the body.
- Score each dimension independently before combining into the final JSON.

Respond ONLY with a valid JSON object, and the STOP. No explanation, no markdown, no extra text.

Valid examples (these are format examples only, not real scores):
{"relevant": false, "Political": null, "Economic": null, "Social": null, "Technological": null}
{"relevant": true, "Political": 0, "Economic": 2, "Social": 0, "Technological": 3}

INVALID — never do this:
{"relevant": true, "Political": null, "Economic": 1, "Social": 0, "Technological": 2}

Respond ONLY with a valid JSON object. No explanation, no markdown, no extra text:
{"relevant":<true/false>, "Political": <0-3>, "Economic": <0-3>, "Social": <0-3>, "Technological": <0-3>}"""


def build_prompt(article: dict) -> str:
    parts = []

    if article.get("Content_Title"):
        parts.append(f"Title: {article['Content_Title']}")
    if article.get("Content"):
        parts.append(f"Content: {article['Content']}")
    if article.get("Content_body"):
        parts.append(f"Body: {article['Content_body']}")

    return "Score the following news article:\n\n" + "\n\n".join(parts)


def has_content(article: dict) -> bool:
    return any(
        article.get(field, "").strip()
        for field in ["Content", "Content_body", "Content_Title"]
    )



def extract_json_object(raw: str) -> dict:
    """Extract the first top-level JSON object from a string, ignoring any
    trailing explanation text the model adds despite instructions not to."""
    # Fast path: maybe it's already clean
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Find the first balanced {...} block
    start = raw.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", raw, 0)

    depth = 0
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(raw[start:i + 1])

    raise json.JSONDecodeError("No balanced JSON object found", raw, start)


def annotate_article(client: Client, article: dict, model: str) -> dict:
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt(article)}
        ]
    )

    raw = response.message.content.strip()

    try:
        result = extract_json_object(raw)
        if "relevant" not in result:
            raise ValueError("Missing 'relevant' field")

        if result["relevant"] is False:
            return {"relevant": False, **{dim: None for dim in DIMENSIONS}}

        if result["relevant"] is True:
            scores = {}
            for dim in DIMENSIONS:
                val = result.get(dim)
                if val is None:
                    val = 0  # per-field null while relevant=true means "no content" → 0
                if val not in (0, 1, 2, 3):
                    raise ValueError(f"Invalid score for {dim}: {val}")
                scores[dim] = val
            return {"relevant": True, **scores}

        raise ValueError(f"Unexpected 'relevant' value: {result['relevant']!r}")

    except (json.JSONDecodeError, ValueError) as e:
        print(f"[Warning] Could not parse response: {e}\nRaw output: {raw}")
        return {"relevant": None, **{dim: None for dim in DIMENSIONS}}

def annotate_dataset(
    input_path: str,
    output_path: str,
    skipped_path: str,
    model: str,
    host: str = "http://localhost:11434"
):
    client = Client(host=host)

    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Transpose column-oriented dict into a list of row dicts
    indices = list(raw["Alert Type"].keys())  # e.g. ["0", "1", "2", ...]
    dataset = [
        {
            "Alert Type": raw["Alert Type"][i],
            "Content": raw["Content"][i],
            "Date": raw["Date"][i],
            "Link": raw["Link"][i],
        }
        for i in indices
    ]

    annotated = []
    skipped = []

    total = len(dataset)
    for idx, article in enumerate(dataset):

        identifier = article.get("link") or article.get("Content_Title") or f"index_{idx}"
        print(f"[{idx + 1}/{total}] Processing: {identifier}")
        if not has_content(article):
            print(f"  -> Skipped (no content found)")
            skipped.append({
                "index": idx,
                "identifier": identifier,
                "reason": "All content fields (Content, Content_body, Content_Title) are empty or missing"
                })
            continue

        scores = annotate_article(client, article, model)

        annotated_article = {
            **article,
            **scores
        }
        annotated.append(annotated_article)


    # Save annotated results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotated, f, indent=2, ensure_ascii=False)

    # Save skipped records
    with open(skipped_path, "w", encoding="utf-8") as f:
        json.dump(skipped, f, indent=2, ensure_ascii=False)

    print(f"\nDone.")
    print(f"  Annotated : {len(annotated)} articles -> {output_path}")
    print(f"  Skipped   : {len(skipped)} articles  -> {skipped_path}")

    return annotated, skipped
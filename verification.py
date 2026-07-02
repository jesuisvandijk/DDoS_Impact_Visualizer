"""
Verification layer for the LLM PESTLE annotations.

This script does NOT re-annotate anything. It takes articles that have
already been scored by `ollama_annotate.py` / `ollama_client.py`
(saved in DATA_FILE) and, for a sample of those articles, asks the model
to justify the score it gave for each dimension by pointing to the
specific part of the article that supports it.

The point is sanity-checking, not re-grading: if the model can't point to
anything concrete in the text to support a score, that's a signal the
annotation might be noise (relevant to the low Cohen's kappa discussed
in the thesis).

Output is a JSON file with, per sampled article and dimension:
    - the original score
    - the model's justification text
    - a short verbatim excerpt the model believes supports the score

Usage:
    python verify_annotations.py
"""

import json
import random

from tqdm import tqdm
from ollama import Client

from article_to_event_level import DATA_FILE
from ollama_client import DIMENSIONS, extract_json_object

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FILE   = DATA_FILE                              # already-annotated articles
OUTPUT_FILE  = "Data/14-verification-results.json"

MODEL        = "llama3.2:1b"
HOST         = "http://localhost:11434"

# How many articles to spot-check. Doesn't need to be huge — this is a
# qualitative sanity check, not a statistical evaluation. ~15 articles
# across 4 dimensions gives 60 justifications, which is enough to spot
# patterns (e.g. the model hand-waving on Social/Political) without being
# a huge extra annotation run.
SAMPLE_SIZE  = 15
RANDOM_SEED  = 42

VERIFY_SYSTEM_PROMPT = """You are a cybersecurity analyst auditing your own previous work.

You will be shown a news article and a score (0-3) that was previously assigned
to it for one PESTLE-style impact dimension. Your job is ONLY to justify or
challenge that score using the article text — you are not assigning a new score.

SCORING SCALE (for reference, this is what the original score means):
0 = No meaningful content relating to this dimension
1 = Minor or indirect mention (topic is peripheral)
2 = Moderate coverage (topic is discussed but not the main focus)
3 = Primary focus (topic is central to the article)

INSTRUCTIONS:
- Point to the specific sentence(s) or fact(s) in the article that justify the score.
- "evidence_quote" must be a short verbatim excerpt from the article (max ~20 words).
  If no such excerpt exists, use an empty string "".
- If you think the score does NOT match the article, say so honestly in "justification"
  and set "agrees_with_score" to false.
- Be concise. Do not repeat the whole article back.

Respond ONLY with a valid JSON object, no markdown, no extra text, no preamble like
"Here is the JSON:" — your entire reply must be parseable by json.loads():
{"agrees_with_score": <true/false>, "justification": "<1-2 sentences>", "evidence_quote": "<short excerpt or empty string>"}
"""

RETRY_USER_SUFFIX = (
    "\n\nYour previous reply could not be parsed as JSON. "
    "Reply again with NOTHING but the raw JSON object, no markdown fences, no extra words."
)


def build_verify_prompt(article: dict, dimension: str, score: int) -> str:
    parts = []
    if article.get("Content_Title"):
        parts.append(f"Title: {article['Content_Title']}")
    if article.get("Content"):
        parts.append(f"Content: {article['Content']}")
    if article.get("Content_body"):
        parts.append(f"Body: {article['Content_body']}")

    article_text = "\n\n".join(parts)

    return (
        f"Dimension: {dimension}\n"
        f"Previously assigned score: {score}\n\n"
        f"Article:\n{article_text}\n\n"
        f"Justify or challenge the above score for the '{dimension}' dimension."
    )


def select_sample(annotated: list, sample_size: int, seed: int = RANDOM_SEED) -> list:
    """Randomly sample only from articles that were marked relevant
    (non-relevant articles have null scores, nothing to verify)."""
    relevant_articles = [a for a in annotated if a.get("relevant") is True]

    if len(relevant_articles) <= sample_size:
        print(f"  Only {len(relevant_articles)} relevant articles available — using all of them.")
        return relevant_articles

    rng = random.Random(seed)
    return rng.sample(relevant_articles, sample_size)


def _parse_verification(raw: str, score) -> dict:
    result = extract_json_object(raw)
    if not isinstance(result, dict):
        raise json.JSONDecodeError("Parsed JSON was not an object", raw, 0)
    return {
        "score": score,
        "agrees_with_score": result.get("agrees_with_score"),
        "justification": result.get("justification", ""),
        "evidence_quote": result.get("evidence_quote", ""),
        "parse_failed": False
    }


def verify_article_dimension(client: Client, article: dict, dimension: str, model: str) -> dict:
    score = article.get(dimension)
    if score is None:
        return None

    prompt = build_verify_prompt(article, dimension, score)
    messages = [
        {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    response = client.chat(
        model=model,
        messages=messages,
        format="json",
        options={"num_ctx": 8192, "num_predict": 400, "temperature": 0}
    )
    raw = response.message.content.strip()

    try:
        return _parse_verification(raw, score)
    except (json.JSONDecodeError, AttributeError):
        pass  # fall through to one retry below

    # Retry once with a stricter instruction, using the conversation so far
    # as context — small models often self-correct when told the format failed.
    messages.append({"role": "assistant", "content": raw})
    messages.append({"role": "user", "content": RETRY_USER_SUFFIX})

    retry_response = client.chat(
        model=model,
        messages=messages,
        format="json",
        options={"num_ctx": 8192, "num_predict": 400, "temperature": 0}
    )
    retry_raw = retry_response.message.content.strip()

    try:
        return _parse_verification(retry_raw, score)
    except (json.JSONDecodeError, AttributeError):
        # Gave up parsing — keep both raw replies so nothing is lost
        return {
            "score": score,
            "agrees_with_score": None,
            "justification": f"[unparsed after retry]\nFirst reply: {raw}\nRetry reply: {retry_raw}",
            "evidence_quote": "",
            "parse_failed": True
        }


def run_verification(
    input_path: str = INPUT_FILE,
    output_path: str = OUTPUT_FILE,
    sample_size: int = SAMPLE_SIZE,
    model: str = MODEL,
    host: str = HOST
):
    print(f"Loading annotated articles from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        annotated = json.load(f)

    print(f"Selecting {sample_size} articles to verify...")
    sample = select_sample(annotated, sample_size)
    print(f"  {len(sample)} articles selected.")

    client = Client(host=host)
    results = []

    for idx, article in enumerate(tqdm(sample, desc="Verifying articles")):
        identifier = article.get("Link") or article.get("Content_Title") or f"index_{idx}"

        dimension_results = {}
        for dimension in DIMENSIONS:
            verification = verify_article_dimension(client, article, dimension, model)
            if verification is not None:
                dimension_results[dimension] = verification

        results.append({
            "identifier": identifier,
            "Content_Title": article.get("Content_Title"),
            "Date": article.get("Date"),
            "Link": article.get("Link"),
            "dimensions": dimension_results
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n_disagreements = sum(
        1
        for r in results
        for d in r["dimensions"].values()
        if d.get("agrees_with_score") is False
    )
    n_parse_failed = sum(
        1
        for r in results
        for d in r["dimensions"].values()
        if d.get("parse_failed") is True
    )
    n_total = sum(len(r["dimensions"]) for r in results)

    print(f"\nDone.")
    print(f"  Verified  : {len(results)} articles, {n_total} dimension-scores -> {output_path}")
    print(f"  Flagged   : {n_disagreements}/{n_total} scores the model itself disagreed with on review")
    print(f"  Unparsed  : {n_parse_failed}/{n_total} replies the model never returned as valid JSON (even after retry)")

    return results


if __name__ == "__main__":
    run_verification()
from ollama_client import annotate_dataset
from article_to_event_level import DATA_FILE

annotate_dataset(
    input_path="Data/test_1000.json",
    output_path= DATA_FILE,
    skipped_path="Data/12-output-pest-skipped.json",
    model="llama3.2:1b",
    host="http://localhost:11434"
)

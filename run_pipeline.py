"""
Runs the full overnight pipeline: annotation -> event-level clustering.
Run this directly in PyCharm (or `python run_pipeline.py` from a terminal).

Why this instead of a .bat file:
- No console encoding issues (Python handles UTF-8 internally here)
- No output buffering surprises
- If annotation fails, the event-level step is skipped automatically
- Progress prints show up immediately, same as running in PyCharm normally
"""

import subprocess
import sys
import datetime

LOG_FILE = "pipeline_log.txt"


def run_step(label: str, script: str) -> bool:
    """Runs a python script as a subprocess, streaming + logging its output live.
    Returns True if it succeeded (exit code 0), False otherwise."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"\n=== [{timestamp}] {label} ==="
    print(header)

    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(header + "\n")
        log.flush()

        # -u = unbuffered, so output streams live instead of arriving in big chunks
        process = subprocess.Popen(
            [sys.executable, "-u", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",  # never crash on a weird character, just substitute it
        )

        # Stream line by line: prints to this console AND writes to the log file
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()

        process.wait()
        return process.returncode == 0


def main():
    # Start with a clean log each run
    with open(LOG_FILE, "w", encoding="utf-8") as log:
        log.write(f"Pipeline started {datetime.datetime.now()}\n")

    print("Step 1: Annotation")
    ok = run_step("Annotation (ollama_annotate.py)", "ollama_annotate.py")

    if not ok:
        print("\nAnnotation failed - aborting before event-level pipeline.")
        print(f"Check {LOG_FILE} for details.")
        sys.exit(1)

    print("\nStep 2: Event-level pipeline")
    ok = run_step("Event-level pipeline (article_to_event_level.py)", "article_to_event_level.py")

    if not ok:
        print("\nEvent-level pipeline failed.")
        print(f"Check {LOG_FILE} for details.")
        sys.exit(1)

    print("\nDone! Full pipeline completed successfully.")


if __name__ == "__main__":
    main()
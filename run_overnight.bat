@echo off
cd /d "C:\Users\JessevanDijk\PycharmProjects\Scriptie"
call .venv\Scripts\activate.bat

echo Starting annotation... > pipeline_log.txt
python ollama_annotate.py >> pipeline_log.txt 2>&1

if errorlevel 1 (
    echo Annotation failed - aborting before event pipeline. >> pipeline_log.txt
    exit /b 1
)

echo Annotation complete. Starting event-level pipeline... >> pipeline_log.txt
python article_to_event_level.py >> pipeline_log.txt 2>&1

echo Done. >> pipeline_log.txt
# Introducion

## Requirements

- [Python 3.7](https://www.python.org/downloads/release/python-379/)
- [ffmpeg](https://ffmpeg.org/)

### Install dependencies

For install requirements use following command:

Activate venv:

```bash
python -m venv venv
```

Install deps:

```bash
pip install -r requirements.txt
```

## Configuration

Use Config from settings.py for adjust processing params

### Main processing params

XLSX_SAVE_PATH - directory for result xlsx

VIDEO_SAVE_FOLDER_PATH - directory for output video

SAVE_FOLDER_PATH - where all frame processing results will be stashed

SAMPLES_DIRECTORY - directory with input sample videos

N2V_MODEL_NAME - model name

N2V_MODEL_DIR - directory with model

SAMPLE_NAME - sample name without extension

SAMPLE_EXTENSION - sample extension 
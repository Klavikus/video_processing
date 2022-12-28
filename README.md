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

### Path params

| Parameter                  | Description                                        |
|----------------------------|----------------------------------------------------|
| **XLSX_SAVE_PATH**         | path for xlsx report                               |
| **VIDEO_SAVE_FOLDER_PATH** | path for output video                              |
| **SAVE_FOLDER_PATH**       | where all frame processing results will be stashed |
| **SAMPLES_DIRECTORY**      | path with input sample videos                      |
| **N2V_MODEL_NAME**         | model name                                         |
| **N2V_MODEL_DIR**          | path with model                                    |
| **SAMPLE_NAME**            | sample name without extension                      |
| **SAMPLE_EXTENSION**       | sample extension                                   |

### Filters params


| Parameter                    | Description                                                                           | Default |
|------------------------------|---------------------------------------------------------------------------------------|---------|
| **NLM_H**                    | Strength                                                                              | 3       |
| **NLM_TEMPLATE_WINDOW_SIZE** | Size in pixels of the template                                                        | 37      |
| **NLM_SEARCH_WINDOW_SIZE**   | Size in pixels of the window that is used to compute weighted average for given pixel | 15      |
| **BILATERAL_D**              | Diameter of each pixel neighborhood that is used during filtering                     | 11      |
| **BILATERAL_SIGMA_COLOR**    | Filter sigma in the color space                                                       | 35      |
| **BILATERAL_SIGMA_SPACE**    | Filter sigma in the coordinate space                                                  | 35      |
| **MEDIAN_K**                 | Aperture linear size                                                                  | 5       |
| **GAUSSIAN_K**               | Kernel size                                                                           | (13,13) |
| **MORPH_OPERATIONS**         | [Link](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)         | -       |

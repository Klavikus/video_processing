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

You can use a pre-created configuration file or create your own based on it. 

Use the following params to customize your own configuration file.

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


| Parameter                    | Description                                                                           | 
|------------------------------|---------------------------------------------------------------------------------------|
| **NLM_H**                    | Strength                                                                              | 
| **NLM_TEMPLATE_WINDOW_SIZE** | Size in pixels of the template                                                        |
| **NLM_SEARCH_WINDOW_SIZE**   | Size in pixels of the window that is used to compute weighted average for given pixel |
| **BILATERAL_D**              | Diameter of each pixel neighborhood that is used during filtering                     |
| **BILATERAL_SIGMA_COLOR**    | Filter sigma in the color space                                                       |
| **BILATERAL_SIGMA_SPACE**    | Filter sigma in the coordinate space                                                  |
| **MEDIAN_K**                 | Aperture linear size                                                                  |
| **GAUSSIAN_K**               | Kernel size                                                                           |
| **MORPH_OPERATIONS**         | [Link](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)         |

## Run

Use the following command to process video file:

```bash
python run.py --config config.toml
```

The following command will invoke video processing using the S1.toml configuration file:

```bash
python run.py --config configs/video_denoise/S1.toml
```

## Train N2V

Use the following command to train the neural network:

```bash
python nn_train.py --config configs.toml
```

The following command will call the neural network training using the base.toml configuration file:

```bash
python nn_train.py --config configs/nn_train/base.toml
```
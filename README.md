# MLT-agent

MLT-agent for AI Foundation coursework

## Overview

This repository contains the **MLT-agent**, a multi-layer transformer-based model designed for Vision-and-Language Navigation (VLN). The agent is built upon the framework provided in [Curriculum-Learning-For-VLN](https://github.com/IMNearth/Curriculum-Learning-For-VLN), with modifications to implement a transformer-based architecture that captures long-range dependencies in both visual and textual inputs. The agent leverages cross-attention mechanisms to align instructions with visual observations and improve navigation performance in unseen environments.

## Repository Structure

The repository includes the following key files:

- `MLT-agent.py`: Contains the implementation of the MLT-agent, including the transformer-based encoder and cross-modal decoder.
- `MLT_config.yaml`: Configuration file for the MLT-agent, defining hyperparameters and model settings.
- `main.py`: The main script to run the navigation task, replacing the original `main.py` from the Curriculum-Learning-For-VLN repository.
- `trainer.py`: Script that handles the training loop for the MLT-agent, replacing the original `trainer.py`.

## Setup Instructions

### 1. Clone two Repository

First, follow the setup instructions from the original repository [Curriculum-Learning-For-VLN](https://github.com/IMNearth/Curriculum-Learning-For-VLN) to set up the simulation environment. Ensure you install all the dependencies and set up the Matterport3D simulator as directed.

```bash
git clone https://github.com/ZhangCXLVII/MLT-agent.git
git clone https://github.com/IMNearth/Curriculum-Learning-For-VLN.git
cd Curriculum-Learning-For-VLN
# Follow the repository instructions to install dependencies and set up the environment
```

### 2. Replace Original Files

After setting up the environment from the Curriculum-Learning-For-VLN repository, replace the following files:

- **Replace `main.py`**: Overwrite the original `main.py` file in `tasks/R2R-judy/` with the `main.py` from this repository.
  
  ```bash
  cp path_to_this_repo/MLT-agent/main.py tasks/R2R-judy/
  ```

- **Replace `trainer.py`**: Overwrite the original `trainer.py` file in `tasks/R2R-judy/src/` with the `trainer.py` from this repository.

  ```bash
  cp path_to_this_repo/MLT-agent/trainer.py tasks/R2R-judy/src/
  ```

- **Add `MLT-agent.py`**: Place the `MLT-agent.py` file in `tasks/R2R-judy/src/agent/`.

  ```bash
  cp path_to_this_repo/MLT-agent/MLT-agent.py tasks/R2R-judy/src/agent/
  ```

- **Add `MLT_config.yaml`**: Place the `MLT_config.yaml` file in `tasks/R2R-judy/configs/MLT-agent/`.

  ```bash
  mkdir -p tasks/R2R-judy/configs/MLT-agent
  cp path_to_this_repo/MLT-agent/MLT_config.yaml tasks/R2R-judy/configs/MLT-agent/
  ```

### 3. Running the MLT-agent

Once you have replaced the files and set up the configuration, you can run the MLT-agent:

```bash
python tasks/R2R-judy/main.py --config-file tasks/R2R-judy/configs/MLT-agent/MLT_config.yaml TRAIN.DEVICE 0
```



## Project Overview

This repository is for the UE23CS352A Machine Learning Mini-Project:  **Learning to Play SLITHER.IO with Deep Reinforcement Learning** . The aim is to build a deep reinforcement learning agent that can play Slither.io, maximizing survival time and snake length using raw gameplay frames as input.

## Problem Statement

* Train a reinforcement learning (RL) agent to play Slither.io by processing raw image frames and outputting action commands (left, right, straight, speed burst).
* The primary objective is to maximize the agent’s survival time and snake length, measuring performance through average score, win rate versus baseline, and score difference compared to random policy.

## Requirements

* Implementation in a single `.ipynb` Jupyter notebook file.
* Use Python with deep RL libraries such as TensorFlow or PyTorch, OpenAI Gym, and Universe for Slither.io environment access.
* The notebook should be self-contained: any requirements should be listed and installed in the first cells using `!pip install` as needed.

## Dataset

* Collect gameplay data from the OpenAI Universe/Gym Slither.io environment or use provided demonstration data.

## Model Design

* Begin with a baseline policy (random or heuristic).
* Develop a Deep Q-Network (DQN) with convolutional layers; optionally explore Double DQN, Dueling DQN, A3C as extensions.
* Reward shaping and prioritized replay are recommended.

## Training & Validation

* Preprocess frames (crop, resize, normalize).
* Use frame skipping to stabilize training (e.g., 5 fps from original 60 fps).
* Track and plot key metrics during training: average episode score, loss curves, comparison with baseline.

## Evaluation

* Present results via plots and Markdown explanations.
* Compare agent performance versus random or human baseline.
* Optional: demo video generation for learned policy.

## Deliverables

* A single `.ipynb` notebook containing:
  * Problem statement, approach, and methodology.
  * Data collection and preprocessing.
  * Model definition and training procedure.
  * Evaluation results, plots, and commentary.
  * Final conclusions and challenges faced.
* All code and explanations should be clear and organized as notebook cells.

## Instructions for Copilot

* Complete all necessary code cells for data loading, model definition, training, and evaluation inside one notebook file.
* Include Markdown cells introducing each major section: Introduction, Data Preprocessing, Model Design, Training Procedure, Results and Evaluation, Conclusions.
* Ensure reproducibility: use random seeds, note package versions, and document hyperparameters.

## How to Run

1. Open `main.ipynb` in JupyterLab or VS Code.
2. Run all cells sequentially (use "Restart & Run All").
3. If any packages are missing, install them in the first cell.
4. Review results in the notebook and refer to markdown explanations for interpretation.

## References

* OpenAI Gym/Universe documentation (for Slither.io environment).
* DQN and RL tutorials.

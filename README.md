# DL - engine

This repository contains two main components for developing a chess AI model:

1. **Supervised Learning Model**  
   A model trained using PGN files and memmap data structures to predict moves on a chess board.

2. **Reinforcement Learning Improvement**  
   A self-play based reinforcement learning script that builds on the pretrained supervised model and improves it using policy gradient methods.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Supervised Training](#supervised-training)
- [Reinforcement Learning Improvement](#reinforcement-learning-improvement)
- [License](#license)

---

## Overview

This project follows a two-stage training approach:

- **Supervised Training:**  
  The model is first trained on historical chess games (in PGN format). Chess positions are converted into 13×8×8 matrices and stored using memmap files. A ResNet-like Convolutional Neural Network is then trained to predict moves. Regular checkpoints are created to enable resuming training if needed.

- **Reinforcement Learning Improvement:**  
  Based on the pretrained model, the reinforcement learning script further improves it through self-play. The model plays chess games against itself, collects the log probabilities of the moves it makes, and uses policy gradient updates to reinforce successful moves. Checkpoints are used to save and load the training state. **I haven't tested this yet. Still under development**

---

## Requirements

- Python 3.7 or higher
- PyTorch
- python-chess
- NumPy
- tqdm

---

## Installation

1. Clone the repository using your preferred Git client.
2. Install the required Python libraries.
3. Place your PGN files in the `engines/torch/data/pgn/` directory. The supervised training script will generate the necessary memmap files and the move mapping file (`move_to_int.pkl`).

---

## Supervised Training

The supervised training script (e.g., `train.py`) performs the following steps:

- **Dataset Creation:**  
  PGN files are used to create memmap files. Each chess position is converted into a 13-channel matrix, and a mapping of possible moves (`move_to_int.pkl`) is generated.

- **Model Setup:**  
  A ResNet-like CNN is defined that accepts 13×8×8 matrices as input and predicts a probability for each possible move.

- **Training:**  
  The model is trained in both training and validation phases. Checkpoints are saved regularly to allow the training to resume if interrupted.

- **Saving:**  
  The final model is saved as `final_chess_model.pth`.

Run the supervised training script to start the training process.

---

## Reinforcement Learning Improvement

The reinforcement learning script (e.g., `reinforce_training.py`) improves the pretrained model using self-play and policy gradient methods:

- **Loading the Model:**  
  The script first loads the pretrained model (`final_chess_model.pth`). If a reinforcement learning checkpoint (`reinforce_checkpoint.pth.tar`) exists, it will load it to resume training.

- **Self-Play:**  
  The model plays complete chess games against itself. For each move, the log probability and the player's role (White or Black) are recorded.

- **Policy Gradient Update:**  
  At the end of a game, a reward is assigned based on the game outcome (win, loss, or draw). The model is then updated using a policy gradient method to reinforce good moves and discourage poor ones.

- **Checkpointing:**  
  After a predetermined number of games, a checkpoint is created that saves both the model and the optimizer state. This allows the training to be resumed later if necessary.

Run the reinforcement learning script to further improve the model through self-play.

---

## License

This project is licensed under the MIT License.

# Chess Vision-Language Model Fine-Tuning (Gemma)

## Overview
This project was developed as part of a university course in Multimodal AI at Hof University of Applied Sciences.  
The goal of the project was to generate chess datasets automatically and fine-tune a vision-language model (Gemma) to understand chess positions and basic chess concepts.

The project focuses on the machine learning pipeline, dataset generation, model fine-tuning using LoRA adapters, and evaluation of the trained model.

---

## Project Goals
The main objectives of this project were:

- Automatically generate chess datasets using chess engines
- Convert chess positions into structured training data
- Fine-tune a vision-language model on chess data
- Train the model in multiple stages
- Evaluate the model on different chess-related tasks
- Analyze strengths and limitations of the model

---

## Training Approach

The model was trained in two stages:

### Stage 1 – Vision Encoder Training
In the first stage, the vision encoder was fine-tuned while the language model remained frozen.  
The goal was to improve the model’s ability to recognize chess pieces, board positions, and spatial relationships on the chessboard.

### Stage 2 – Language Model Training
In the second stage, the vision encoder was frozen and the language model was fine-tuned.  
The goal was to improve reasoning about chess positions, such as material evaluation, board descriptions, and basic chess understanding.

---

## Results
The model showed improvements in several areas:

**Improved tasks**
- Piece identification
- Board description tasks
- Material evaluation
- Game outcome prediction
- Structured output formatting

**Difficult tasks**
- Move prediction
- Tactical reasoning
- Complex chess strategies
- Attack detection

A detailed evaluation and analysis of the training results can be found in the following report:

Finetuning_Eval_Results.pdf

---

## Technologies Used

- Python
- PyTorch
- HuggingFace Transformers
- PEFT / LoRA
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebooks
- Stockfish chess engine
- Machine Learning / Deep Learning
- Vision-Language Models

---

## Machine Learning Pipeline

The overall workflow of the project:

1. Generate chess games using chess engines
2. Extract board positions from games
3. Convert positions into training datasets
4. Train vision encoder (Stage 1)
5. Train language model (Stage 2)
6. Evaluate model performance
7. Analyze results and limitations

---

## How to Run (General Steps)

1. Install Python dependencies:

pip install -r requirements.txt


2. Generate datasets:

python data_generation/games_generation.py


3. Run training notebooks:

notebooks/stage1_vision_training.ipynb
notebooks/stage2_llm_training.ipynb


4. Run evaluation:

python stage2_final_eval.py


---

## Project Background
This project was created as part of a university course and focuses on building a machine learning pipeline, dataset generation, model fine-tuning, and evaluation rather than building a complete chess engine.

The project demonstrates:
- Dataset generation
- Data preprocessing
- Model fine-tuning
- Machine learning experimentation
- Evaluation and analysis
- Working with large models
- Research-style project workflow

---

## Author
Emrah Zehic  
Computer Science Student  
Hof University of Applied Sciences

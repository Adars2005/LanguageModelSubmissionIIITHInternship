Neural Language Model Training using PyTorch

Submitted by:
J. Adarsh
B.Tech CSD, Keshav Memorial Institute of Technology (KMIT), Hyderabad
Email: adarshjangeeti@gmail.com

🔍 Objective

This project implements a Neural Language Model (NLM) completely from scratch using PyTorch, demonstrating how sequence models learn to predict text and how model capacity, dropout, and regularization impact generalization.

Three training regimes were implemented as required:

Underfitting — very small model → high bias

Overfitting — large model + no dropout → memorization

Best Fit — balanced model → optimal generalization

Evaluation metrics used:

Cross-Entropy Loss

Perplexity (PPL)

📘 Dataset & Preprocessing

Dataset: Pride and Prejudice by Jane Austen (public domain)
Size: ~700 KB (≈130k tokens)

Preprocessing Pipeline

Custom word-level tokenizer (whitespace-based)

Special tokens: <pad>, <unk>, <bos>, <eos>

Vocabulary size: ~25,000 tokens

Sliding window sequence creation (seq_len = 20–30)

90% Train / 10% Validation split

Custom LangModelDataset + PyTorch DataLoader

⚙️ Model Architecture

Implemented a 2-layer LSTM Language Model:

Component	Description
Embedding	Token → vector (emb_size)
LSTM	Learns sequential dependencies
Dropout	Prevents overfitting
Linear	Hidden → vocabulary logits
Loss	CrossEntropyLoss
Optimizer	Adam (lr = 1e-3)
Metric	Perplexity = exp(loss)

Random seed fixed (SEED = 42) for reproducibility.

🧪 Experimental Configurations
Config	Hidden	Layers	Dropout	Batch	Epochs	LR	Behavior
Underfit	32	1	0.5	128	6	0.005	Too small → fails to learn
Overfit	512	2	0.0	16	20	0.001	Large → memorizes
Best Fit	256	2	0.2	64	12	0.001	Best generalization
🖥️ Training Setup
Parameter	Value
Runtime	Google Colab (CPU)
Framework	PyTorch 2.x
Device	cpu
Tokens	~150k
Avg Time	Underfit: 13s/epoch • Overfit: 147s/epoch • Best Fit: 49s/epoch
📊 Results
Final Metrics
Model	Final Train Loss	Final Val Loss	Val Perplexity	Notes
Underfit	28.7 → 47.9	40 → 47	~10¹⁸–10²⁰	Model failed to learn
Overfit	1.21 → 0.13	5.25 → 9.49	~13k	Memorization
Best Fit	11.26 → 2.21	12.37 → 17.57	~4×10⁷	Balanced learning
Training Curve Interpretation

Underfit: flat, high losses

Overfit: train ↓ while val ↑

Best Fit: stable downward trend → best-generalizing model

📈 Analysis & Interpretation

Underfitting caused by insufficient model capacity

Overfitting caused by zero dropout & large hidden size

Best Fit demonstrates correct bias–variance trade-off

Weight decay + dropout improved generalization

Gradient clipping stabilized training

🧮 Perplexity Definition
Perplexity
=
𝑒
CrossEntropyLoss
Perplexity=e
CrossEntropyLoss

Lower PPL indicates better next-token prediction.

📂 Repository Structure
LanguageModelSubmissionIIITHInternship/
│
├── LanguageModel.ipynb            # Main notebook
├── language_model.py              # Training/Model scripts
├── Pride_and_Prejudice-Jane_Austen.txt
│
├── lm_best_fit.pt                 # Best Fit model
├── lm_underfit.pt                 # Underfit model
# lm_overfit.pt excluded due to >25 MB GitHub limit
│
└── Assignment 2 — Language Model using PyTorch.pdf

🧾 Notes on Model Files

GitHub restricts uploads >25MB.
Therefore:

lm_underfit.pt → uploaded

lm_best_fit.pt → uploaded

lm_overfit.pt → excluded but can be shared via Google Drive if required

▶️ How to Run

Clone the repo:

git clone https://github.com/Adars2005/LanguageModelSubmissionIIITHInternship
cd LanguageModelSubmissionIIITHInternship


Install dependencies:

pip install torch numpy matplotlib


Open and run the notebook:

Upload the dataset when prompted.

Run all cells sequentially.

Models will be saved in the working directory.

Reproducibility:

Random seeds fixed (SEED = 42)

Runs produce identical results.

🔗 References

Bengio et al. (2003), A Neural Probabilistic Language Model

Goodfellow et al. (2016), Deep Learning — Chapter 10

PyTorch Docs — https://pytorch.org/docs

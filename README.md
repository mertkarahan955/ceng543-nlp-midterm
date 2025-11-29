# CENG543 NLP Midterm Project

This project contains the NLP midterm assignment for CENG543 course. All questions can be run with a single command.

## Quick Start

### 1. Environment Setup

First, set up the Python environment:

```bash
./setup_environment.sh
```

This script offers two options:
- **Conda** (Recommended): Better dependency management
- **pip + virtualenv**: Simpler, doesn't require conda

### 2. Activate Environment

**If you used Conda:**
```bash
conda activate ceng543-nlp
```

**If you used pip:**
```bash
source venv/bin/activate
```

### 3. Run All Questions

```bash
./run_all_questions.sh
```

or for a specific question only:

```bash
./run_all_questions.sh q1   # Question 1 only
./run_all_questions.sh q2   # Question 2 only
./run_all_questions.sh q3   # Question 3 only
./run_all_questions.sh q4   # Question 4 only
./run_all_questions.sh q5   # Question 5 only
```

## Requirements

### System Requirements
- Python 3.8 or higher
- CUDA-capable GPU (recommended, not mandatory)
- Minimum 8GB RAM
- Minimum 20GB disk space

### Python Packages

All required packages are listed in `requirements.txt` and `environment.yml`:

- **Deep Learning**: PyTorch, Transformers
- **NLP**: Datasets, Tokenizers, Sentence-Transformers
- **Word Embeddings**: Gensim
- **Metrics**: SacreBLEU, ROUGE, BERTScore
- **ML**: scikit-learn, NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Explainability**: LIME

## Project Structure

```
.
├── ceng543_m_q1/           # Question 1: Text Classification
├── ceng543_m_q2/           # Question 2: Attention Mechanisms
├── ceng543_m_q3/           # Question 3: Transformer Architecture
├── ceng543_m_q4/           # Question 4: RAG Systems
├── ceng543_m_q5/           # Question 5: Model Interpretation
├── requirements.txt        # pip requirements
├── environment.yml         # conda environment
├── setup_environment.sh    # Environment setup script
└── run_all_questions.sh    # Main execution script
```

## Manual Installation

If you prefer not to use the automatic setup script:

### With Conda:
```bash
conda env create -f environment.yml
conda activate ceng543-nlp
```

### With pip:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Features

- Run all questions with a single command
- Automatic environment validation
- Colored terminal output
- Progress tracking and duration calculation
- Automatic CUDA/GPU detection
- Error handling and safe exit

## Outputs

Each question generates results in its own directory:
- **Model checkpoints**: Trained models for each question
- **Metrics**: Metrics in JSON format
- **Visualizations**: Graphs in PNG format
- **Logs**: Training and evaluation logs

## Notes

- First run will download datasets and models from HuggingFace
- GPU usage is recommended, otherwise training will be very slow
- Running all questions may take several hours
- Ensure you have sufficient disk space

## Troubleshooting

### "CUDA out of memory" error:
- Reduce batch size
- Use `--no-cuda` parameter in some scripts

### Import errors:
```bash
# Reinstall environment
./setup_environment.sh
```

### Script not running:
```bash
# Check execute permissions
chmod +x setup_environment.sh
chmod +x run_all_questions.sh
```

## License

This project was prepared for the CENG543 course.

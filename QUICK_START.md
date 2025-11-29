# Quick Start Guide

## Step 1: Environment Setup

```bash
./setup_environment.sh
```

Choose from the menu:
- **1** → Conda (Recommended)
- **2** → pip + virtualenv
- **3** → Skip if already installed

## Step 2: Activate Environment

**If you chose Conda:**
```bash
conda activate ceng543-nlp
```

**If you chose pip:**
```bash
source venv/bin/activate
```

## Step 3: Run All Questions

```bash
./run_all_questions.sh
```

This command will:
- Check environment
- Run all questions sequentially
- Show progress
- Calculate total time

---

## To Run a Single Question:

```bash
./run_all_questions.sh q1   # Question 1 only
./run_all_questions.sh q2   # Question 2 only
./run_all_questions.sh q3   # Question 3 only
./run_all_questions.sh q4   # Question 4 only
./run_all_questions.sh q5   # Question 5 only
```

---

## One-Line Setup + Run

```bash
./setup_environment.sh && conda activate ceng543-nlp && ./run_all_questions.sh
```

---

## Troubleshooting

### If you get import errors:
```bash
pip install -r requirements.txt
```

### If scripts don't run:
```bash
chmod +x setup_environment.sh run_all_questions.sh
```

### For detailed information:
```bash
cat README.md
```

---

**Note:** First run will download datasets from HuggingFace. Make sure you have an active internet connection!

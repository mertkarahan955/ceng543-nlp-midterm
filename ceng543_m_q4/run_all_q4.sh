#!/bin/bash

echo "=========================================="
echo "Q4: RAG System - Complete Pipeline"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Corpus: Wikipedia"
echo "  Sparse Retrieval: TF-IDF"
echo "  Dense Retrieval: Sentence-BERT"
echo "  Generator: FLAN-T5-base"
echo ""
echo "=========================================="
echo ""

START_TIME=$(date +%s)

echo "[Step 1/10] Preparing Wikipedia dataset..."
python 1_prepare_data.py
if [ $? -ne 0 ]; then
    echo "Error in data preparation. Exiting."
    exit 1
fi
echo ""

echo "[Step 2/10] Building TF-IDF retriever..."
python 2_build_bm25_retriever.py
if [ $? -ne 0 ]; then
    echo "Error in TF-IDF indexing. Exiting."
    exit 1
fi
echo ""

echo "[Step 3/10] Building Sentence-BERT retriever..."
python 3_build_sbert_retriever.py
if [ $? -ne 0 ]; then
    echo "Error in SBERT indexing. Exiting."
    exit 1
fi
echo ""

echo "[Step 4/10] Evaluating retrieval systems..."
python 4_evaluate_retrieval.py
if [ $? -ne 0 ]; then
    echo "Error in retrieval evaluation. Exiting."
    exit 1
fi
echo ""

echo "[Step 5/10] Running oracle generation (gold passages)..."
python 5_oracle_generation.py
if [ $? -ne 0 ]; then
    echo "Error in oracle generation. Exiting."
    exit 1
fi
echo ""

echo "[Step 6/10] Running TF-IDF + FLAN-T5-base RAG pipeline..."
python 6_rag_bm25.py
if [ $? -ne 0 ]; then
    echo "Error in TF-IDF RAG pipeline. Exiting."
    exit 1
fi
echo ""

echo "[Step 7/10] Running SBERT + FLAN-T5-base RAG pipeline..."
python 7_rag_sbert.py
if [ $? -ne 0 ]; then
    echo "Error in SBERT RAG pipeline. Exiting."
    exit 1
fi
echo ""

echo "[Step 8/10] Evaluating generation quality..."
python 8_evaluate_generation.py
if [ $? -ne 0 ]; then
    echo "Error in generation evaluation. Exiting."
    exit 1
fi
echo ""

echo "[Step 9/10] Generating qualitative examples..."
python 9_qualitative_analysis.py
if [ $? -ne 0 ]; then
    echo "Error in qualitative analysis. Exiting."
    exit 1
fi
echo ""

echo "[Step 10/10] Creating visualizations..."
python 10_visualize_results.py
if [ $? -ne 0 ]; then
    echo "Error in visualization. Exiting."
    exit 1
fi
echo ""

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo "=========================================="
echo "✅ Q4 PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results saved in:"
echo "  - data/                   (preprocessed dataset)"
echo "  - models/                 (TF-IDF & SBERT indexes)"
echo "  - outputs/                (all results & metrics)"
echo "  - outputs/plots/          (visualizations)"
echo ""
echo "Key files:"
echo "  - outputs/retrieval_metrics.json"
echo "  - outputs/generation_metrics.json"
echo "  - outputs/qualitative_examples.md"
echo ""
echo "=========================================="
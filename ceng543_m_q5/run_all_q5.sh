#!/bin/bash

echo "=========================================="
echo "Q5: Interpretability & Error Analysis"
echo "=========================================="
echo ""
echo "Selected Model: Transformer + DistilBERT (Q3)"
echo "BLEU: 6.49 | ROUGE-L: 0.6355"
echo ""
echo "=========================================="
echo ""

START_TIME=$(date +%s)

echo "[Step 1/7] Loading best model from Q3..."
python 1_load_best_model.py
if [ $? -ne 0 ]; then
    echo "Error loading model. Exiting."
    exit 1
fi
echo ""

echo "[Step 2/7] Generating attention visualizations..."
python 2_attention_visualization.py
if [ $? -ne 0 ]; then
    echo "Error in attention visualization. Exiting."
    exit 1
fi
echo ""

echo "[Step 3/7] Computing Integrated Gradients..."
python 3_integrated_gradients.py
if [ $? -ne 0 ]; then
    echo "Error in Integrated Gradients. Exiting."
    exit 1
fi
echo ""

echo "[Step 4/7] Running LIME analysis..."
python 4_lime_analysis.py
if [ $? -ne 0 ]; then
    echo "Error in LIME analysis. Exiting."
    exit 1
fi
echo ""

echo "[Step 5/7] Analyzing failure cases..."
python 5_failure_case_analysis.py
if [ $? -ne 0 ]; then
    echo "Error in failure analysis. Exiting."
    exit 1
fi
echo ""

echo "[Step 6/7] Quantifying uncertainty..."
python 6_uncertainty_quantification.py
if [ $? -ne 0 ]; then
    echo "Error in uncertainty quantification. Exiting."
    exit 1
fi
echo ""

echo "[Step 7/7] Creating summary visualizations..."
python 7_visualize_results.py
if [ $? -ne 0 ]; then
    echo "Error in visualization. Exiting."
    exit 1
fi
echo ""

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo "=========================================="
echo "✅ Q5 PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "Total runtime: ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results saved in:"
echo "  - outputs/attention_heatmaps/     (5 examples + multihead)"
echo "  - outputs/integrated_gradients/   (3 examples)"
echo "  - outputs/lime_explanations/      (3 examples + comparison)"
echo "  - outputs/failure_cases.json      (5 categorized failures)"
echo "  - outputs/uncertainty_metrics.json"
echo "  - outputs/summary/q5_dashboard.png"
echo ""
echo "Key findings:"
echo "  - Attention: Diagonal patterns in translation"
echo "  - Failure modes: OOV, long-distance, negation, coreference, idioms"
echo "  - Uncertainty: Higher entropy for incorrect predictions"
echo ""
echo "=========================================="
# CENG543 Q1 - Comparative Analysis of Recurrent Architectures and Embedding Paradigms

This project implements a comparative analysis of bidirectional LSTM and GRU architectures with GloVe and BERT embeddings on the ATIS (Intent Detection) dataset.

## Dataset

- **ATIS**: Airline Travel Information System dataset for intent detection (multiple intent classes)

## Usage

### Running all experiments

```bash
# Run all experiments for ATIS
./run_all_q1.sh atis

# Or without argument (default is atis)
./run_all_q1.sh
```

### Manual training

**GloVe + BiLSTM:**
```bash
python src/train.py --dataset atis --mode glove --rnn_type lstm --epochs 5 --batch_size 64 --out_dir models/atis_glove_lstm
```

**GloVe + BiGRU:**
```bash
python src/train.py --dataset atis --mode glove --rnn_type gru --epochs 5 --batch_size 64 --out_dir models/atis_glove_gru
```

**BERT + BiLSTM:**
```bash
python src/train.py --dataset atis --mode bert --rnn_type lstm --freeze_bert --epochs 3 --batch_size 16 --out_dir models/atis_bert_lstm
```

**BERT + BiGRU:**
```bash
python src/train.py --dataset atis --mode bert --rnn_type gru --freeze_bert --epochs 3 --batch_size 16 --out_dir models/atis_bert_gru
```

### Extracting embeddings

```bash
python src/extract_embeddings.py --dataset atis --mode glove --rnn_type lstm --model_path models/atis_glove_lstm/best_atis_glove_lstm.pt --out_dir outputs --out_name atis_glove_lstm_emb
```

### Visualizing embeddings

```bash
python src/visualize_embeddings.py --emb_npz outputs/atis_glove_lstm_emb.npz --out_dir outputs/emb_viz --prefix atis_glove_lstm
```

### Evaluating models

```bash
python src/evaluate_f1.py --dataset atis --mode glove --model_path models/atis_glove_lstm/best_atis_glove_lstm.pt --rnn_type lstm
```

## Requirements

- GloVe embeddings should be in `~/.vector_cache/glove.6B.300d.word2vec.txt` (word2vec format)
- All Python dependencies are listed in `requirements.txt`

## Output Structure

- `models/atis_{mode}_{rnn}/`: Trained models and training logs
- `outputs/atis_{mode}_{rnn}_emb.npz`: Extracted embeddings
- `outputs/emb_viz/atis_{mode}_{rnn}_pca.png`: PCA visualization
- `outputs/emb_viz/atis_{mode}_{rnn}_tsne.png`: t-SNE visualization

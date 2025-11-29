# src/visualize_embeddings.py
import argparse, os, numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_2d(z, labels, outpath, title):
    plt.figure(figsize=(10,10))
    # Use colormap for multiple classes
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    if num_classes <= 10:
        # Use distinct colors for small number of classes
        colors = plt.cm.tab10(np.linspace(0, 1, max(10, num_classes)))
    else:
        # Use colormap for many classes
        colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
    
    for i, lab in enumerate(unique_labels):
        idx = labels == lab
        plt.scatter(z[idx,0], z[idx,1], s=6, alpha=0.6, label=f'Class {int(lab)}', c=[colors[i]])
    plt.legend(title='label', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_npz', required=True, help='.npz with arrays embeddings,labels')
    parser.add_argument('--out_dir', default='outputs/emb_viz')
    parser.add_argument('--prefix', default='emb')
    parser.add_argument('--pca_dim', type=int, default=50)
    parser.add_argument('--perplexity', type=int, default=50)
    args = parser.parse_args()

    data = np.load(args.emb_npz)
    X = data['embeddings']
    y = data['labels'].astype(int)
    os.makedirs(args.out_dir, exist_ok=True)

    # PCA (2 bileşen)
    pca2 = PCA(n_components=2)
    X_pca2 = pca2.fit_transform(X)
    plot_2d(X_pca2, y, os.path.join(args.out_dir, f'{args.prefix}_pca.png'), f'PCA: {args.prefix}')

    # PCA ile boyutu indirip t-SNE
    if args.pca_dim and X.shape[1] > args.pca_dim:
        X_reduced = PCA(n_components=args.pca_dim).fit_transform(X)
    else:
        X_reduced = X
    tsne = TSNE(
    n_components=2,
    perplexity=args.perplexity,
    init='pca',
    learning_rate='auto',
    max_iter=1000
)
    X_tsne = tsne.fit_transform(X_reduced)
    plot_2d(X_tsne, y, os.path.join(args.out_dir, f'{args.prefix}_tsne.png'), f't-SNE: {args.prefix}')

    print("Saved visualizations to", args.out_dir)

if __name__ == '__main__':
    main()

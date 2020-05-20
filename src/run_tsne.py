import argparse
import numpy as np
from tools.utils import get_logger
from sklearn.manifold import TSNE
from tools.data_io import load_object, save_object

logger = get_logger('tSNE')


def main():
    parser = argparse.ArgumentParser(description='Load a saved pca object')
    parser.add_argument('--low-dim-bin-path', type=str)
    parser.add_argument('--save-bin-path', type=str)
    parser.add_argument('--num-pca-component', type=int, default=10)
    parser.add_argument('--dim-embedded', type=int, default=2)
    args = parser.parse_args()

    logger.info(f'Load low dim data from {args.low_dim_bin_path}')
    low_dim_array = load_object(args.low_dim_bin_path)
    data_matrix = np.zeros((len(low_dim_array), args.num_pca_component))
    for sample_idx in range(len(low_dim_array)):
        data_matrix[sample_idx, :] = low_dim_array[sample_idx]['low_dim'][:]

    logger.info(f'Num of sample: {data_matrix.shape[0]}')
    logger.info(f'Num of included PCs: {data_matrix.shape[1]}')

    logger.info('Start tSNE')
    # embedded_matrix = TSNE(perplexity=50, learning_rate=10000, n_components=args.dim_embedded).fit_transform(data_matrix)
    embedded_matrix = TSNE(perplexity=50, learning_rate=100000, n_components=args.dim_embedded).fit_transform(
        data_matrix)
    logger.info('Complete')
    logger.info(f'Output shape: {embedded_matrix.shape}')

    for sample_idx in range((len(low_dim_array))):
        low_dim_array[sample_idx]['tsne_data'] = embedded_matrix[sample_idx, :]

    # logger.info(low_dim_array[0])

    logger.info(f'Save data to {args.save_bin_path}')
    save_object(low_dim_array, args.save_bin_path)


if __name__ == '__main__':
    main()
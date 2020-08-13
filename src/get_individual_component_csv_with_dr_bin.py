import argparse
from tools.pca import PCA_NII_3D
from tools.data_io import ScanWrapper, DataFolder, save_object, load_object
from tools.utils import get_logger
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

logger = get_logger('PCA low dimension')


def main():
    parser = argparse.ArgumentParser(description='Load a saved pca object')
    parser.add_argument('--load-dr-bin', type=str)
    parser.add_argument('--component-id', type=int)
    parser.add_argument('--save-bin-png', type=str)
    parser.add_argument('--save-csv', type=str)
    args = parser.parse_args()

    dr_data = load_object(args.load_dr_bin)

    file_list = dr_data['file_list']
    projected_data = dr_data['projected_matrix']

    component_val_array = projected_data[:, args.component_id - 1]

    component_dict_array = [{
        'Scan': file_list[file_idx],
        'Value': component_val_array[file_idx]
    } for file_idx in range(len(file_list))]

    plt.hist(
        component_val_array
    )

    plt.savefig(args.save_bin_png)

    df = pd.DataFrame(component_dict_array)
    logger.info(f'Save csv to {args.save_csv}')
    df.to_csv(args.save_csv, index=False)


if __name__ == '__main__':
    main()
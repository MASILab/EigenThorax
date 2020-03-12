import argparse
from tools.config import load_config
from tools.preprocess import ScanFolderFlatReader
from tools.pca import PCA_NII_3D
from tools.utils import mkdir_p


def main():
    parser = argparse.ArgumentParser(description='Run preprocess on in data folder')
    parser.add_argument('--config', type=str,
                        help='Path to the YAML config file', required=True)
    parser.add_argument('--in-folder', type=str,
                        help='Folder of input data', required=True)
    parser.add_argument('--ref-img', type=str,
                        help='Use the average as reference image', required=True)
    parser.add_argument('--out-pc-folder', type=str,
                        help='Output location of principle images', required=True)
    parser.add_argument('--out-mean-img', type=str,
                        help='Output the mean of pca', required=True)
    parser.add_argument('--n-components', type=int,
                        help='Number of pc', required=True)

    args = parser.parse_args()

    config = load_config(args.config)

    scan_folder_reader = ScanFolderFlatReader(config, args.in_folder, args.ref_img)
    scan_folder_reader.read_data()

    pca_nii_3d = PCA_NII_3D(scan_folder_reader, args.ref_img, args.n_components)
    pca_nii_3d.run_pca()
    mkdir_p(args.out_pc_folder)
    pca_nii_3d.write_pc(args.out_pc_folder)
    pca_nii_3d.write_mean(args.out_mean_img)


if __name__ == '__main__':
    main()
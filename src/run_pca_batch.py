import argparse
from tools.config import load_config
from tools.preprocess import ScanFolderBatchReader
from tools.pca import PCA_NII_3D_Batch
from tools.utils import mkdir_p


def main():
    parser = argparse.ArgumentParser(description='Run preprocess on in data folder')
    parser.add_argument('--config', type=str,
                        help='Path to the YAML config file', required=True)
    parser.add_argument('--in-folder', type=str,
                        help='Folder of input data', required=True)
    parser.add_argument('--in-file-list', type=str,
                        help='file list to use in pca')
    parser.add_argument('--ref-img', type=str,
                        help='Use the average as reference image', required=True)
    parser.add_argument('--out-pc-folder', type=str,
                        help='Output location of principle images')
    parser.add_argument('--out-mean-img', type=str,
                        help='Output the mean of pca', required=True)
    parser.add_argument('--n-components', type=int,
                        help='Number of pc', required=True)
    parser.add_argument('--n-batch', type=int,
                        help='Number of batches')
    parser.add_argument('--save-pca-result-path', type=str,
                        help='Save the result to the given location')

    args = parser.parse_args()

    config = load_config(args.config)

    scan_folder_reader = ScanFolderBatchReader(config,
                                               args.in_folder,
                                               args.ref_img,
                                               args.n_batch,
                                               file_list_txt=args.in_file_list)

    pca_nii_3d = PCA_NII_3D_Batch(scan_folder_reader, args.ref_img, args.n_components, args.n_batch)
    pca_nii_3d.run_pca()

    if args.save_pca_result_path:
        pca_nii_3d.save_pca_obj(args.save_pca_result_path)

    if args.out_pc_folder:
        mkdir_p(args.out_pc_folder)
        pca_nii_3d.write_pc(args.out_pc_folder)
    if args.out_mean_img:
        pca_nii_3d.write_mean(args.out_mean_img)


if __name__ == '__main__':
    main()
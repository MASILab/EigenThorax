import argparse
from tools.config import load_config
from tools.preprocess import ScanFolderConcatBatchReaderWithMask
from tools.pca import PCA_NII_3D_Batch
from tools.utils import mkdir_p


def main():
    parser = argparse.ArgumentParser(description='Run preprocess on in data folder')
    parser.add_argument('--config', type=str,
                        help='Path to the YAML config file', required=True)
    parser.add_argument('--in-ori-folder', type=str,
                        help='Folder of input data', required=True)
    parser.add_argument('--in-jac-folder', type=str, required=True)
    parser.add_argument('--mask-img-path', type=str, required=True)
    parser.add_argument('--in-file-list', type=str,
                        help='file list to use in pca')
    parser.add_argument('--out-pc-folder', type=str,
                        help='Output location of principle images')
    parser.add_argument('--n-components', type=int,
                        help='Number of pc', required=True)
    parser.add_argument('--batch-size', type=int,
                        help='Batch size')
    parser.add_argument('--save-pca-result-path', type=str,
                        help='Save the result to the given location')
    parser.add_argument('--load-pca-obj-file-path', type=str)

    args = parser.parse_args()

    config = load_config(args.config)

    scan_folder_reader = ScanFolderConcatBatchReaderWithMask(
        args.mask_img_path,
        config,
        args.in_ori_folder,
        args.in_jac_folder,
        args.batch_size,
        file_list_txt=args.in_file_list)

    pca_nii_3d = PCA_NII_3D_Batch(scan_folder_reader,
                                  None,
                                  args.n_components)

    if args.load_pca_obj_file_path:
        pca_nii_3d.load_pca(args.load_pca_obj_file_path)
    else:
        pca_nii_3d.run_pca()

        if args.save_pca_result_path:
            pca_nii_3d.save_pca_obj(args.save_pca_result_path)

    if args.out_pc_folder:
        mkdir_p(args.out_pc_folder)
        pca_nii_3d.write_pc(args.out_pc_folder)


if __name__ == '__main__':
    main()
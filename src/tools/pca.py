from tools.data_io import ScanWrapper
from sklearn.decomposition import PCA
import time


class PCA_NII_3D:
    def __init__(self, scan_folder_reader, ref_img, n_components):
        self._ref_img = ScanWrapper(ref_img)
        self._scan_folder_reader = scan_folder_reader
        self._n_components = n_components
        self._pc = []
        self._mean = []

    def run_pca(self):
        data_matrix = self._scan_folder_reader.get_data_matrix()

        print(f'Run PCA with {self._n_components} components')
        print(f'Input matrix shape is {data_matrix.shape}')

        tic = time.perf_counter()
        pca = PCA(n_components=self._n_components)
        pca.fit(data_matrix)
        toc = time.perf_counter()
        print(f'Done. Total time on pca: {toc - tic:0.4f} (s)')
        self._pc = pca.components_
        self._mean = pca.mean_
        print(f'Output pc matrix shape is {self._pc.shape}')
        print(f'Output mean shape is {self._mean.shape}')

    def write_pc(self, out_folder):
        print(f'Save pcs to {out_folder}')
        for idx_pc in range(self._n_components):
            out_path = f'{out_folder}/pc_{idx_pc}.nii.gz'
            pc_data_flat = self._pc[idx_pc, :]
            self._ref_img.save_scan_flat_img(pc_data_flat, out_path)

    def write_mean(self, out_path):
        print('Save mean')
        self._ref_img.save_scan_flat_img(self._mean, out_path)







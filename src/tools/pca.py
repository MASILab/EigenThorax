from tools.data_io import ScanWrapper
from sklearn.decomposition import PCA
import time
from tools.data_io import save_object, load_object


class PCA_NII_3D:
    def __init__(self, scan_folder_reader, ref_img, n_components):
        if ref_img:
            self._ref_img = ScanWrapper(ref_img)
        self._scan_folder_reader = scan_folder_reader
        self._pca = PCA(n_components=n_components)

    def load_pca(self, bin_path):
        print(f'Loading pca from ${bin_path}', flush=True)
        self._pca = load_object(bin_path)

    def run_pca(self):
        data_matrix = self._scan_folder_reader.get_data_matrix()

        n_components = self._pca.n_components
        print(f'Run PCA with {n_components} components', flush=True)
        print(f'Input matrix shape is {data_matrix.shape}', flush=True)

        tic = time.perf_counter()
        self._pca.fit(data_matrix)
        toc = time.perf_counter()
        print(f'Done. Total time on pca: {toc - tic:0.4f} (s)', flush=True)
        print(f'Output pc matrix shape is {self._pca.components_.shape}', flush=True)
        print(f'Output mean shape is {self._pca.mean_.shape}', flush=True)

    def write_pc(self, out_folder):
        print(f'Save pcs to {out_folder}', flush=True)
        n_components = self._pca.n_components
        pc = self._pca.components_
        for idx_pc in range(n_components):
            out_path = f'{out_folder}/pc_{idx_pc}.nii.gz'
            pc_data_flat = pc[idx_pc, :]
            self._ref_img.save_scan_flat_img(pc_data_flat, out_path)

    def write_mean(self, out_path):
        print('Save mean', flush=True)
        mean = self._pca.mean_
        self._ref_img.save_scan_flat_img(mean, out_path)

    def get_scree_array(self):
        return self._pca.explained_variance_ratio_

    def save_pca_obj(self, file_path):
        save_object(self._pca, file_path)

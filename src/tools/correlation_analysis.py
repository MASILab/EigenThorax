from tools.utils import get_logger
import os
import numpy as np
import matplotlib.pyplot as plt
from tools.data_io import ClusterAnalysisDataDict
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import seaborn as sns
import matplotlib.tri as tri
from tools.regression import EigenThoraxLinearRegression1D
from tools.lda import EigenThoraxLDA1D
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE


logger = get_logger('Correlation Analysis')


class CorrelationAnalysis2OrthoSpace:
    def __init__(self, data_dict_obj: ClusterAnalysisDataDict):
        self._data_obj = data_dict_obj

    def plot_2D_grid_pack_field_tsne_list(self, out_png_folder):
        fig, ax = plt.subplots(figsize=(20, 14))
        gs = gridspec.GridSpec(2, 3)

        ax_list = []
        for idx_ax in range(6):
            ax_list.append(plt.subplot(gs[idx_ax]))

        self._plot_2D_tsne_embeding_field_continue(fig, ax_list[0], 'bmi')
        self._plot_2D_tsne_embeding_field_continue(fig, ax_list[1], 'Age')
        self._plot_2D_tsne_embeding_field_continue(fig, ax_list[2], 'Packyear')
        self._plot_2D_tsne_embedding_field_discrect(fig, ax_list[3], 'CAC')
        self._plot_2D_tsne_embedding_field_discrect(fig, ax_list[4], 'COPD')
        self._plot_2D_tsne_embedding_field_discrect(fig, ax_list[5], 'CancerSubjectFirstScan')

        out_png = os.path.join(out_png_folder, '2d_tsne_embedded_plot.png')
        logger.info(f'Save to {out_png}')
        fig.tight_layout()
        plt.savefig(out_png)
        plt.close()

    def _plot_2D_tsne_embeding_field_continue(self, fig, ax, field_flag):
        data_X, data_Y = self._data_obj.get_features_and_labels(field_flag)
        tsne_embedded_matrix = self._get_tsne_2d_embeding_data(data_X)

        scatter_points = ax.scatter(
            tsne_embedded_matrix[:, 0],
            tsne_embedded_matrix[:, 1],
            c=data_Y[:],
            cmap='jet'
        )

        ax.set_title(f'{field_flag}')

        fig.colorbar(scatter_points, ax=ax)
        ax.set_aspect('auto')
        ax.set_xlabel('t-SNE embedded 2D space, dimension 1')
        ax.set_ylabel('t-SNE embedded 2D space, dimension 2')

    def _plot_2D_tsne_embedding_field_discrect(self, fig, ax, field_flag):
        data_X, data_Y = self._data_obj.get_features_and_labels(field_flag)
        tsne_embedded_matrix = self._get_tsne_2d_embeding_data(data_X)

        field_label_list = self._data_obj._get_field_label_list(field_flag)

        for idx_class in range(np.max(data_Y.astype(np.int)) + 1):
            idx_list_class_label = np.where(data_Y == idx_class)
            ax.scatter(
                tsne_embedded_matrix[idx_list_class_label, 0],
                tsne_embedded_matrix[idx_list_class_label, 1],
                label=field_label_list[idx_class]
            )

        ax.set_title(f'{field_flag}')
        ax.legend(loc='best')
        ax.set_xlabel('t-SNE embedded 2D space, dimension 1')
        ax.set_ylabel('t-SNE embedded 2D space, dimension 2')

    def _get_tsne_2d_embeding_data(self, data_matrix):
        logger.info('Start tSNE')
        embedded_matrix = TSNE(perplexity=50, n_iter=1000, n_components=2, random_state=768).fit_transform(
            data_matrix)
        logger.info('Complete')
        return embedded_matrix

    def plot_2D_grid_pack_field_list(self, out_png_folder):
        fig, ax = plt.subplots(figsize=(20, 14))
        gs = gridspec.GridSpec(2, 3)

        ax_list = []
        for idx_ax in range(6):
            ax_list.append(plt.subplot(gs[idx_ax]))

        self._plot_2D_top_dim_ortho_ax(fig, ax_list[0], 'bmi')
        self._plot_2D_top_dim_ortho_ax(fig, ax_list[1], 'Age')
        self._plot_2D_top_dim_ortho_ax(fig, ax_list[2], 'Packyear')
        self._plot_2D_top_dim_lda_ortho_ax(fig, ax_list[3], 'CAC')
        self._plot_2D_top_dim_lda_ortho_ax(fig, ax_list[4], 'COPD')
        self._plot_2D_top_dim_lda_ortho_ax(fig, ax_list[5], 'CancerSubjectFirstScan')

        out_png = os.path.join(out_png_folder, '2d_plot_top_correlation.png')
        logger.info(f'Save to {out_png}')
        fig.tight_layout()
        plt.savefig(out_png)
        plt.close()

    def _plot_2D_top_dim_ortho_ax(self, fig, ax, field_flag):
        data_X, data_Y = self._data_obj.get_features_and_labels(field_flag)
        dominant_space_2d = self._get_top_2_dim_projector_regression(field_flag)

        scatter_points = ax.scatter(
            dominant_space_2d[:, 0],
            dominant_space_2d[:, 1],
            c=data_Y[:],
            cmap='jet'
        )

        ax.set_title(f'Top 2 correlation axes, {field_flag}')

        fig.colorbar(scatter_points, ax=ax)
        ax.set_aspect('auto')

    def _plot_2D_top_dim_lda_ortho_ax(self, fig, ax, field_flag):
        data_X, data_Y = self._data_obj.get_features_and_labels(field_flag)
        dominant_space_2d = self._get_top_2_dim_projector_LDA(field_flag)

        field_label_list = self._data_obj._get_field_label_list(field_flag)

        for idx_class in range(np.max(data_Y.astype(np.int)) + 1):
            idx_list_class_label = np.where(data_Y == idx_class)
            ax.scatter(
                dominant_space_2d[idx_list_class_label, 0],
                dominant_space_2d[idx_list_class_label, 1],
                label=field_label_list[idx_class]
            )

        ax.set_title(f'Top 2 correlation axes, {field_flag}')
        ax.legend(loc='best')

    def plot_2D_top_dim_ortho(self, field_flag, out_png_folder):
        data_X, data_Y = self._data_obj.get_features_and_labels(field_flag)
        dominant_space_2d = self._get_top_2_dim_projector_regression(field_flag)

        fig, ax = plt.subplots(figsize=(20, 14))
        scatter_points = ax.scatter(
            dominant_space_2d[:, 0],
            dominant_space_2d[:, 1],
            c=data_Y[:],
            cmap='jet'
        )

        plt.title(f'Top 2 correlation axes, {field_flag}')
        fig.colorbar(scatter_points)

        out_png = os.path.join(out_png_folder, f'ortho_top_2_dim_{field_flag}.png')
        logger.info(f'Save to {out_png}')
        plt.savefig(out_png)
        plt.close()

    def plot_2D_top_dim_lda_ortho(self, field_flag, out_png_folder):
        data_X, data_Y = self._data_obj.get_features_and_labels(field_flag)
        dominant_space_2d = self._get_top_2_dim_projector_LDA(field_flag)

        field_label_list = self._data_obj._get_field_label_list(field_flag)

        fig, ax = plt.subplots(figsize=(20, 14))

        for idx_class in range(np.max(data_Y.astype(np.int)) + 1):
            idx_list_class_label = np.where(data_Y == idx_class)
            ax.scatter(
                dominant_space_2d[idx_list_class_label, 0],
                dominant_space_2d[idx_list_class_label, 1],
                label=field_label_list[idx_class]
            )

        plt.title(f'Top 2 correlation axes, {field_flag}')
        plt.legend(loc='best')

        out_png = os.path.join(out_png_folder, f'ortho_top_2_dim_{field_flag}.png')
        logger.info(f'Save to {out_png}')
        plt.savefig(out_png)
        plt.close()


    def _get_top_2_dim_projector_regression(self, field_flag):
        n_feature = self._data_obj.get_num_feature()
        data_X, data_Y = self._data_obj.get_features_and_labels(field_flag)

        # dominant_space_2d = np.zeros((self._data_obj.get_num_sample(), 2))
        dominant_space_2d = np.zeros((data_X.shape[0], 2))
        linear_reg_obj = EigenThoraxLinearRegression1D(data_X, data_Y)
        linear_reg_obj.run_regression()
        dominant_space_2d[:, 0] = linear_reg_obj.project_to_dominant_space()[:, 0]
        projected_data_X = linear_reg_obj.project_to_complement_space()

        linear_reg_obj_2 = EigenThoraxLinearRegression1D(projected_data_X, data_Y)
        linear_reg_obj_2.run_regression()
        dominant_space_2d[:, 1] = linear_reg_obj_2.project_to_dominant_space()[:, 0]

        return dominant_space_2d

    def _get_top_2_dim_projector_LDA(self, field_flag):
        n_feature = self._data_obj.get_num_feature()
        data_X, data_Y = self._data_obj.get_features_and_labels(field_flag)

        dominant_space_2d = np.zeros((data_X.shape[0], 2))
        lda_obj = EigenThoraxLDA1D(data_X, data_Y)

        dominant_space_2d[:, 0] = lda_obj.project_to_dominant_space()[:, 0]
        projected_data_X = lda_obj.project_to_complement_space()

        lda_obj_2 = EigenThoraxLDA1D(projected_data_X, data_Y)
        dominant_space_2d[:, 1] = lda_obj_2.project_to_dominant_space()[:, 0]

        return dominant_space_2d


class CorrelationAnalysis:
    def __init__(self, data_dict_obj : ClusterAnalysisDataDict):
        self._data_obj = data_dict_obj

    def correlation_bar_plot(self, field_flag, output_folder):
        n_feature = self._data_obj.get_num_feature()
        data_X, data_Y = self._data_obj.get_features_and_labels(field_flag)

        corr_val_list = np.zeros((n_feature,), dtype=float)

        for idx_feature in range(n_feature):
            corr_val_list[idx_feature] = pearsonr(data_X[:, idx_feature], data_Y[:])[0]

        fig = plt.figure(figsize=(20, 14))
        plt.bar(range(n_feature), corr_val_list)
        plt.title(f'Pearson Correlation, {field_flag}')

        out_png = os.path.join(output_folder, f'corr_{field_flag}.png')
        logger.info(f'Save to {out_png}')
        plt.savefig(out_png)
        plt.close()

        return corr_val_list.argsort()[-2:]

    def mutual_info_bar_plot(self, field_flag, output_folder):
        n_feature = self._data_obj.get_num_feature()
        data_X, data_Y = self._data_obj.get_features_and_labels(field_flag)

        mutual_info_vec = mutual_info_regression(data_X, data_Y)

        fig = plt.figure(figsize=(20, 14))
        plt.bar(range(n_feature), mutual_info_vec)
        plt.title(f'Mutual information, {field_flag}')

        out_png = os.path.join(output_folder, f'mutual_info_{field_flag}.png')
        logger.info(f'Save to {out_png}')
        plt.savefig(out_png)
        plt.close()

    def plot_2D_dim_plot(self, idx_2_top, field_flag, out_png_folder):
        idx_dim1 = idx_2_top[0]
        idx_dim2 = idx_2_top[1]

        n_feature = self._data_obj.get_num_feature()
        data_X, data_Y = self._data_obj.get_features_and_labels(field_flag)

        # cmap = sns.cubehelix_palette(as_cmap=True)

        fig, ax = plt.subplots(figsize=(20, 14))

        # triangled = tri.Triangulation(data_X[:, idx_dim1], data_X[:, idx_dim2])
        #
        # ax.tricontourf(triangled, data_Y[:])

        scatter_points = ax.scatter(
            data_X[:, idx_dim1],
            data_X[:, idx_dim2],
            c=data_Y[:],
            cmap='jet'
        )

        plt.title(f'Top 2 correlation axes, {field_flag}')
        fig.colorbar(scatter_points)

        out_png = os.path.join(out_png_folder, f'2D_top_dim_{field_flag}.png')
        logger.info(f'Save to {out_png}')
        plt.savefig(out_png)
        plt.close()



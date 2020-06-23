import numpy as np
from tools.utils import get_logger
from scipy.linalg import null_space
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


logger = get_logger('LDA')


class EigenThoraxLDA1D:
    def __init__(self, data_X, data_Y):
        self._data_X = data_X
        self._data_Y = data_Y
        self._lda_obj = LinearDiscriminantAnalysis(n_components=1)
        self._lda_obj.fit(data_X, data_Y)

    def project_to_complement_space(self):
        dominant_space = np.zeros((1, self._lda_obj.coef_.shape[1]))
        dominant_space[0, :] = self._lda_obj.coef_[0, :]
        logger.info(f'Calculate null space of space with shape {dominant_space.shape}')
        null_space_corr = null_space(dominant_space)
        logger.info(f'Get null space with shape {null_space_corr.shape}')
        logger.info(f'Project data_X (with shape {self._data_X.shape}) into null space')
        projected_data_X = np.matmul(self._data_X, null_space_corr)
        logger.info(f'Projected data_X with shape {projected_data_X.shape}')

        return projected_data_X

    def project_to_dominant_space(self):
        dominant_space = np.zeros((self._lda_obj.coef_.shape[1], 1))
        dominant_space[:, 0] = self._lda_obj.coef_[0, :]

        projected_data_X = np.matmul(self._data_X, dominant_space)

        return projected_data_X

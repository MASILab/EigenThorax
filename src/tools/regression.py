import numpy as np
from tools.utils import get_logger
from sklearn.linear_model import LinearRegression
from scipy.linalg import null_space


logger = get_logger('Regression')


class EigenThoraxLinearRegression1D:
    def __init__(self, data_X, data_Y):
        self._data_X = data_X
        self._data_Y = data_Y
        self._linear_reg = LinearRegression()

    def run_regression(self):
        logger.info('Run 1D linear regression on EigenThorax space')
        self._linear_reg.fit(self._data_X, self._data_Y)
        r2_score = self._linear_reg.score(self._data_X, self._data_Y)
        logger.info(f'The R^2 score is {r2_score}')

    def project_to_complement_space(self):
        bmi_space = np.zeros((1, len(self._linear_reg.coef_)))
        bmi_space[0, :] = self._linear_reg.coef_[:]
        logger.info(f'Calculate null space of bmi_space with shape {bmi_space}')
        null_space_corr = null_space(bmi_space)
        logger.info(f'Get null space with shape {null_space_corr.shape}')
        logger.info(f'Project data_X (with shape {self._data_X.shape}) into null space')
        projected_data_X = np.matmul(self._data_X, null_space_corr)
        logger.info(f'Projected data_X with shape {projected_data_X.shape}')

        return projected_data_X
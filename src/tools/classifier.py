import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tools.preprocess import ScanFolderBatchReader
from tools.utils import get_logger
from tools.cross_validation import get_idx_list_array_n_fold_cross_validation
from sklearn import metrics


logger = get_logger('Classifier')


def get_validation_statics(label, predicted_prob):
    fpr, tpr, _ = metrics.roc_curve(label, predicted_prob, pos_label=1)
    precision, recall, _ = metrics.precision_recall_curve(label, predicted_prob, pos_label=1)
    roc_auc = metrics.roc_auc_score(label, predicted_prob)
    prc_auc = metrics.auc(recall, precision)

    summary_item = {
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'prc_auc': prc_auc,
        'label': label,
        'pred': predicted_prob
    }

    return summary_item


class MinibatchLinearClassifierWithCV:
    def __init__(
            self,
            num_fold,
            train_data_folder_obj_list,
            valid_data_folder_obj_list, label_dict):
        self.num_fold = num_fold

        self.model_list = []
        for idx_fold in range(num_fold):
            fold_model = MinibatchLinearClassifierSingleFold(
                train_data_folder_obj_list[idx_fold],
                valid_data_folder_obj_list[idx_fold],
                label_dict
            )
            self.model_list.append(fold_model)

        self.validation_result = []

    def train_first_fold(self):
        logger.info(f'Start to train model of fold 1')
        self.model_list[0].train()

    def valid_first_fold(self):
        logger.info(f'Run validation with the first fold')
        perf_statics = self.model_list[0].validate()
        self.validation_result.append(perf_statics)

    def train(self):
        for idx_model in range(len(self.model_list)):
            logger.info(f'Start to train model of fold {idx_model}')
            self.model_list[idx_model].train()

    def validate(self):
        for idx_model in range(len(self.model_list)):
            logger.info(f'Start to validate model of fold {idx_model}')
            perf_statics = self.model_list[idx_model].validate()
            self.validation_result.append(perf_statics)

    def show_cross_validation_result(self, out_folder):
        for idx_fold in range(len(self.validation_result)):
            logger.info(f'Show validation result of fold {idx_fold}')
            print(self.validation_result[idx_fold])

    @staticmethod
    def create_classifier_obj(in_folder, file_list, num_fold, in_label_df, batch_size):
        """
        Create classifier obj with cross-validation.
        :param in_folder:
        :param file_list:
        :param num_fold:
        :param in_label_df: two column ['scan', 'label']
        :param batch_size:
        :return:
        """
        label_list = in_label_df['label'].to_list()
        fold_train_idx_list_array, fold_test_idx_list_array = \
            get_idx_list_array_n_fold_cross_validation(file_list, label_list, num_fold)

        train_data_folder_obj_list = []
        valid_data_folder_obj_list = []
        label_dict = dict(zip(file_list, label_list))
        for idx_fold in range(num_fold):
            fold_train_idx_list = fold_train_idx_list_array[idx_fold]
            fold_test_idx_list = fold_test_idx_list_array[idx_fold]

            train_file_list = [file_list[idx] for idx in fold_train_idx_list]
            valid_file_list = [file_list[idx] for idx in fold_test_idx_list]

            train_data_folder_obj = ScanFolderBatchReader(
                config=None,
                in_folder=in_folder,
                ref_img=None,
                batch_size=batch_size,
                file_list=train_file_list
            )
            valid_data_folder_obj = ScanFolderBatchReader(
                config=None,
                in_folder=in_folder,
                ref_img=None,
                batch_size=batch_size,
                file_list=valid_file_list
            )

            train_data_folder_obj_list.append(train_data_folder_obj)
            valid_data_folder_obj_list.append(valid_data_folder_obj)

        classifier_obj = MinibatchLinearClassifierWithCV(
            num_fold,
            train_data_folder_obj_list,
            valid_data_folder_obj_list,
            label_dict
        )

        return classifier_obj


class MinibatchLinearClassifierSingleFold:
    def __init__(self, train_data_folder_obj, valid_data_folder_obj, label_dict):
        self.num_epoch = 1
        self.train_data_folder_obj = train_data_folder_obj
        self.valid_data_folder_obj = valid_data_folder_obj
        self.label_dict = label_dict
        self.model = None

    def train(self):
        num_batch = self.train_data_folder_obj.num_batch()
        logger.info(f'Number of batch: {num_batch}')

        self.model = SGDClassifier(n_jobs=-1, loss='log')
        for idx_epoch in range(self.num_epoch):
            logger.info(f'Start epoch {idx_epoch}')
            for idx_batch in range(num_batch):
                self.train_data_folder_obj.read_data(idx_batch)
                data_matrix = self.train_data_folder_obj.get_data_matrix()
                batch_file_name_list = self.train_data_folder_obj.get_batch_file_name_list(idx_batch)
                batch_label = self.get_file_label_list(batch_file_name_list)

                self.model.partial_fit(data_matrix, batch_label, classes=np.array([0, 1]))

    def validate(self):
        pred_prob_list = []
        true_label_list = []
        for idx_batch in range(self.valid_data_folder_obj.num_batch()):
            self.valid_data_folder_obj.read_data(idx_batch)
            data_X = self.valid_data_folder_obj.get_data_matrix()
            batch_pred_prob = self.model.predict_proba(data_X)
            pred_prob_list.append(batch_pred_prob)

            batch_file_name_list = self.valid_data_folder_obj.get_batch_file_name_list(idx_batch)
            batch_label = self.get_file_label_list(batch_file_name_list)
            true_label_list.append(batch_label)

        pred_prob_list = np.concatenate(tuple(pred_prob_list))
        true_label_list = np.concatenate(tuple(true_label_list))
        pred_prob_list = pred_prob_list[:, 1]
        print(pred_prob_list.shape)
        print(true_label_list.shape)

        validation_result = get_validation_statics(true_label_list, pred_prob_list)

        return validation_result

    def get_file_label_list(self, file_list):
        label_list = np.array([self.label_dict[file_name] for file_name in file_list])
        return label_list

import argparse
from tools.pca import PCA_NII_3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.ticker import MaxNLocator
from tools.clinical import ClinicalDataReaderSPORE
from tools.data_io import load_object
from tools.utils import get_logger
import pandas as pd
import os


logger = get_logger('Plot - PC 2 dim')


class PlotSpacePCA:
    def __init__(self, label_df):
        self._label_df = label_df
        self._x_lim = (-2500, 2500)
        self._y_lim = (-5000, 5000)
        self._if_set_lim = False

    def save_label_file(self, out_csv):
        self._label_df.to_csv(out_csv)

    def plot_copd(self, out_png):
        plt.figure(figsize=(16, 10))

        df_yes = self._label_df[self._label_df['copd'] == 'Yes']
        df_no = self._label_df[self._label_df['copd'] == 'No']
        df_unknown = self._label_df[pd.isnull(self._label_df['copd'])]

        logger.info(f'df_yes {df_yes.shape[0]}')
        logger.info(f'df_no {df_no.shape[0]}')
        logger.info(f'df_unknown {df_unknown.shape[0]}')

        ax = plt.figure(figsize=(16, 7))
        plt.title('COPD')
        plt.scatter(
            x=df_yes['pc0'],
            y=df_yes['pc1'],
            c='red',
            alpha=0.3,
            label=f'copd:Yes ({df_yes.shape[0]})'
        )
        plt.scatter(
            x=df_no['pc0'],
            y=df_no['pc1'],
            c='blue',
            alpha=0.3,
            label=f'copd:No ({df_no.shape[0]})'
        )

        ax.legend()
        if self._if_set_lim:
            plt.xlim(left=self._x_lim[0], right=self._x_lim[1])
            plt.ylim(bottom=self._y_lim[0], top=self._y_lim[1])
        logger.info(f'Save to {out_png}')
        plt.savefig(out_png)

    def plot_age(self, out_png):
        plt.figure(figsize=(16, 10))

        df_1 = self._label_df[self._label_df['Age'] <= 60]
        df_2 = self._label_df[self._label_df['Age'] >= 70]
        df_3 = self._label_df[(self._label_df['Age'] > 60) & (self._label_df['Age'] < 70)]

        logger.info(f'df_1 {df_1.shape[0]}')
        logger.info(f'df_2 {df_2.shape[0]}')
        logger.info(f'df_3 {df_3.shape[0]}')

        ax = plt.figure(figsize=(16, 7))
        plt.title('Age')
        plt.scatter(
            x=df_1['pc0'],
            y=df_1['pc1'],
            c='red',
            alpha=0.3,
            label=f'Age <= 60 ({df_1.shape[0]})'
        )
        plt.scatter(
            x=df_2['pc0'],
            y=df_2['pc1'],
            c='blue',
            alpha=0.3,
            label=f'Age >= 70 ({df_2.shape[0]})'
        )
        # plt.scatter(
        #     x=df_3['pc0'],
        #     y=df_3['pc1'],
        #     c='gray',
        #     alpha=0.3
        # )

        ax.legend()
        if self._if_set_lim:
            plt.xlim(left=self._x_lim[0], right=self._x_lim[1])
            plt.ylim(bottom=self._y_lim[0], top=self._y_lim[1])
        logger.info(f'Save to {out_png}')
        plt.savefig(out_png)

    def plot_packyear(self, out_png):
        plt.figure(figsize=(16, 10))


        df_1 = self._label_df[self._label_df['packyearsreported'] <= 35]
        df_2 = self._label_df[self._label_df['packyearsreported'] >= 60]
        df_unknown = self._label_df[(self._label_df['packyearsreported'] > 35) & (self._label_df['packyearsreported'] < 60)]

        logger.info(f'df_yes {df_1.shape[0]}')
        logger.info(f'df_no {df_2.shape[0]}')
        logger.info(f'df_unknown {df_unknown.shape[0]}')

        ax = plt.figure(figsize=(16, 7))
        plt.title('packyear')
        plt.scatter(
            x=df_1['pc0'],
            y=df_1['pc1'],
            c='red',
            alpha=0.3,
            label=f'packyear <= 35 ({df_1.shape[0]})'
        )
        plt.scatter(
            x=df_2['pc0'],
            y=df_2['pc1'],
            c='blue',
            alpha=0.3,
            label=f'packyear >= 60 ({df_2.shape[0]})'
        )
        # plt.scatter(
        #     x=df_unknown['pc0'],
        #     y=df_unknown['pc1'],
        #     c='gray',
        #     alpha=0.3
        # )

        ax.legend()
        if self._if_set_lim:
            plt.xlim(left=self._x_lim[0], right=self._x_lim[1])
            plt.ylim(bottom=self._y_lim[0], top=self._y_lim[1])
        logger.info(f'Save to {out_png}')
        plt.savefig(out_png)

    def plot_ca_cal(self, out_png):

        sub_df_list = []
        label_types = {
            'Severe': {
                'color': 'orange',
                'title': 'Severe'
            },
            'Moderate': {
                'color': 'red',
                'title': 'Moderate'
            },
            'Mild': {
                'color': 'green',
                'title': 'Mild'
            },
            'None': {
                'color': 'blue',
                'title': 'None'
            }
        }

        ax = plt.figure(figsize=(16, 7))
        plt.title('Coronary Artery Calcification')
        for label in label_types:
            sub_df = self._label_df[self._label_df['Coronary Artery Calcification'] == label]
            sub_df_list.append(sub_df)
            num_sample = sub_df.shape[0]
            logger.info(f'{label}: {num_sample}')

            plt.scatter(
                x=sub_df['pc0'],
                y=sub_df['pc1'],
                c=label_types[label]['color'],
                alpha=0.3,
                label=f'{label}: {num_sample}'
            )

        ax.legend()
        if self._if_set_lim:
            plt.xlim(left=self._x_lim[0], right=self._x_lim[1])
            plt.ylim(bottom=self._y_lim[0], top=self._y_lim[1])
        logger.info(f'Save to {out_png}')
        plt.savefig(out_png)

    def plot_bmi(self, out_png):

        label_types = {
            'Low': {
                'color': 'orange',
                'title': 'Low BMI (<21)',
                'sub_df': self._label_df[self._label_df['bmi'] < 21]
            },
            'High': {
                'color': 'green',
                'title': 'High BMI (>35)',
                'sub_df': self._label_df[self._label_df['bmi'] > 35]
            }
        }

        ax = plt.figure(figsize=(16, 7))
        plt.title('BMI')
        for label in label_types:
            sub_df = label_types[label]['sub_df']
            num_sample = sub_df.shape[0]
            logger.info(f'{label}: {num_sample}')

            title = label_types[label]['title']
            plt.scatter(
                x=sub_df['pc0'],
                y=sub_df['pc1'],
                c=label_types[label]['color'],
                alpha=0.3,
                label=f'{title}: {num_sample}'
            )

        ax.legend()
        if self._if_set_lim:
            plt.xlim(left=self._x_lim[0], right=self._x_lim[1])
            plt.ylim(bottom=self._y_lim[0], top=self._y_lim[1])
        logger.info(f'Save to {out_png}')
        plt.savefig(out_png)

    @staticmethod
    def create_class_object_w_data(data_array, label_obj, low_dim_data_flag):
        data_dict = {}
        attribute_list = PlotSpacePCA.attribute_list()
        for data_item in data_array:
            item_dict = {}
            scan_name = data_item['scan_name']

            if not label_obj.check_if_have_record(scan_name):
                logger.info(f'Cannot find record for {scan_name}')
                continue

            for attr in attribute_list:
                item_dict[attr] = label_obj.get_value_field(scan_name, attr)

            # BMI = mass(lb)/height(inch)^2 * 703
            bmi_val = np.nan
            mass_lb = item_dict['weightpounds']
            height_inch = item_dict['heightinches']
            if (70 < mass_lb < 400) and (40 < height_inch < 90):
                bmi_val = 703 * mass_lb / (height_inch * height_inch)
            item_dict['bmi'] = bmi_val

            for pc_idx in range(2):
                attr_str = PlotSpacePCA.get_pc_str(pc_idx)
                item_dict[attr_str] = data_item[low_dim_data_flag][pc_idx]

            data_dict[scan_name] = item_dict

        df = pd.DataFrame.from_dict(data_dict, orient='index')
        print(df)

        return PlotSpacePCA(df)

    @staticmethod
    def create_class_object_w_csv(csv_path):
        logger.info(f'Load csv data file from {csv_path}')
        df = pd.read_csv(csv_path)
        return PlotSpacePCA(df)

    @staticmethod
    def attribute_list():
        return [
            'Age', 'sex', 'race', 'ctscannermake', 'heightinches',
            'weightpounds', 'packyearsreported', 'copd', 'Coronary Artery Calcification',
        ]

    @staticmethod
    def get_pc_str(idx):
        return f'pc{idx}'


def main():
    parser = argparse.ArgumentParser(description='Load a saved pca object')
    parser.add_argument('--in-data-bin', type=str)
    parser.add_argument('--out-png-folder', type=str)
    parser.add_argument('--label-file', type=str)
    parser.add_argument('--data-csv', type=str, default=None)
    parser.add_argument('--low-dim-data-flag', type=str, default='low_dim')
    args = parser.parse_args()

    plot_obj = None
    if args.data_csv is not None:
        plot_obj = PlotSpacePCA.create_class_object_w_csv(args.data_csv)
    else:
        low_dim_array = load_object(args.in_data_bin)
        label_obj = ClinicalDataReaderSPORE.create_spore_data_reader_xlsx(args.label_file)
        plot_obj = PlotSpacePCA.create_class_object_w_data(low_dim_array, label_obj, args.low_dim_data_flag)
        out_csv = os.path.join(args.out_png_folder, 'data.csv')
        plot_obj.save_label_file(out_csv)

    plot_obj.plot_copd(os.path.join(args.out_png_folder, 'copd.png'))
    plot_obj.plot_age(os.path.join(args.out_png_folder, 'age.png'))
    plot_obj.plot_packyear(os.path.join(args.out_png_folder, 'packyear.png'))
    plot_obj.plot_ca_cal(os.path.join(args.out_png_folder, 'ca_cal.png'))
    plot_obj.plot_bmi(os.path.join(args.out_png_folder, 'bmi.png'))

if __name__ == '__main__':
    main()
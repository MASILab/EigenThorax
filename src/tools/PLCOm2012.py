import numpy as np
from tools.clinical import ClinicalDataReaderSPORE
from tools.utils import get_logger
import pandas as pd

SPORE_label_csv = '/nfs/masi/SPORE/file/clinical/self_creat/Limitedhistory20200420.csv'

logger = get_logger('PLCOm2012')


def PLCOm2012(age, race, education, body_mass_index, copd, phist, fhist,
              smoking_status, smoking_intensity, duration, quit_time):  # this for spore, please also refer to Norm_cancer_risk
    def get_num(x):
        if x == 'yes' or x == 'current' or x == 1:
            return 1
        else:
            return 0
    def get_race(x):
        d = {1.0: 0, 2.0: 0.3944778,
             2.5: -0.7434744, 3.0: -0.466585,
             4.0: 0,
             5.0: 1.027152}
        return d[x]
    age_item = 0.0778868 * (age - 62)
    edu_item = -0.0812744 * (education - 4)
    bmi_item = -0.0274194 * (body_mass_index - 27)
    copd_item = 0.3553063 * get_num(copd)
    phist_item = 0.4589971 * get_num(phist)
    fhist_item = 0.587185 * get_num(fhist)
    sstatus_item = 0.2597431 * (smoking_status - 1)
    sint_item = - 1.822606 * (10 / smoking_intensity - 0.4021541613)
    duration_item = 0.0317321 * (duration - 27)
    qt_item = -0.0308572 * (quit_time - 10)
    res = age_item + get_race(race) + edu_item + bmi_item \
          +copd_item + phist_item + fhist_item + sstatus_item \
          + sint_item + duration_item + qt_item - 4.532506
    res = np.exp(res) / (1 + np.exp(res))
    return res

def _get_PLCOm2012_score_csv():
    # df = pd.read_csv('/nfs/masi/SPORE/file/clinical/self_creat/Limitedhistory20200420.csv', index_col='sess')
    df = pd.read_csv('/nfs/masi/xuk9/SPORE/clustering/registration/20200512_corrField/male/clinical/label_full.csv', index_col='id')
    # plco_list = []
    # for i, item in df.iterrows():
    #     if item['age'] == item['age'] and item['race'] == item['race'] and item['education'] == item['education'] and \
    #             item['bmi'] == item['bmi'] and item['copd'] == item['copd'] and item['phist'] == item['phist'] and item[
    #         'fhist'] == item['fhist'] and item['smo_status'] == item['smo_status'] and item['smo_intensity'] == item[
    #         'smo_intensity'] and item['duration'] == item['duration'] and item['quit_time'] == item['quit_time']:
    #         plco = PLCOm2012(age=item['age'], race=item['race'], education=item['education'],
    #                          body_mass_index=item['bmi'], copd=item['copd'], phist=item['phist'], fhist=item['fhist'],
    #                          smoking_status=item['smo_status'], smoking_intensity=item['smo_intensity'],
    #                          duration=item['duration'], quit_time=item['quit_time'])
    #         plco_list.append(plco)
    #     else:
    #         plco_list.append('')
    # df['plco'] = plco_list

    return df

def get_PLCOm2012_score(file_list):
    df = _get_PLCOm2012_score_csv()
    df = df.replace(np.nan, '', regex=True)

    # df.set_index('sess')
    index_list = df.index.to_list()
    # print(index_list[:10])
    index_list_unique = list(set(index_list))
    # print(f'Num unique sess before clean: {len(index_list_unique)}')
    index_list_unique = [sess for sess in index_list_unique if str(sess) != 'nan']
    # print(f'Num unique sess after clean: {len(index_list_unique)}')
    idx_list = [index_list.index(sess) for sess in index_list_unique]
    df_cleaned = df.ix[idx_list]
    # logger.info(f'Num full index: {len(index_list)}')
    # logger.info(f'Num unique index: {len(index_list_unique)}')
    data_dict = df_cleaned.to_dict('index')

    # print(list(data_dict.keys())[:5])

    file_list_without_ext = [file_name.replace('.nii.gz', '') for file_name in file_list]

    # Check if missing sess
    missing_list_sess = [sess for sess in file_list_without_ext if sess not in data_dict]
    logger.info(f'Number of missing of sessions: {len(missing_list_sess)}')
    print(missing_list_sess)
    missing_list_plco = [sess for sess in file_list_without_ext if (sess not in missing_list_sess) and (data_dict[sess]['plco'] == '')]
    logger.info(f'Number of missing of PLCOm: {len(missing_list_plco)}')
    missing_list = list(set(missing_list_sess + missing_list_plco))
    logger.info(f'Total number of missing: {len(missing_list)}')

    valid_idx_list = [idx for idx, sess in enumerate(file_list_without_ext) if sess not in missing_list]

    PLCOm2012_score_list = [data_dict[sess]['plco'] for sess in file_list_without_ext if sess not in missing_list]

    logger.info(f'Size of sore: {len(PLCOm2012_score_list)}')
    print(PLCOm2012_score_list[:10])
    logger.info(f'Size of valid: {len(valid_idx_list)}')

    return PLCOm2012_score_list, valid_idx_list


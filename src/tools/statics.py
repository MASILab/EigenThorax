import pandas as pd
import numpy as np
from scipy.stats import chi2
from tools.utils import get_logger


logger = get_logger('Statics')


def hl_test(data, g, label_column, prob_column='prob'):
    '''
    Hosmer-Lemeshow test to judge the goodness of fit for binary data

    Input: dataframe(data), integer(num of subgroups divided)

    Output: float
    '''
    data_st = data.sort_values(prob_column)
    data_st['dcl'] = pd.qcut(data_st[prob_column], g)

    ys = data_st[label_column].groupby(data_st.dcl).sum()
    yt = data_st[label_column].groupby(data_st.dcl).count()
    yn = yt - ys

    yps = data_st[prob_column].groupby(data_st.dcl).sum()
    ypt = data_st[prob_column].groupby(data_st.dcl).count()
    ypn = ypt - yps

    hltest = (((ys - yps) ** 2 / yps) + ((yn - ypn) ** 2 / ypn)).sum()
    pval = 1 - chi2.cdf(hltest, g - 2)

    df = g - 2

    print(ys)
    print(yps)
    print('\n HL-chi2({}): {}, p-value: {}\n'.format(df, hltest, pval))
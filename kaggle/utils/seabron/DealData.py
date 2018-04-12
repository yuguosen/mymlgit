# coding=utf-8
import pandas as pd
import numpy as np
from sklearn import preprocessing


# 数据预处理的一些基本类

# 根据数据的缺失率，删除缺失率大于miss_rate_filter的列
def deal_miss_rate(df, miss_rate_filter):
    # 求出每一列空值的数目
    miss_total = df.isnull().sum().sort_values(ascending=False)
    miss_pct = (df.isnull().sum() / df.isnull().count().sort_values(ascending=False)).sort_values(ascending=False)
    miss_data = pd.concat([miss_total, miss_pct], axis=1, keys=["Total", "Percent"]).sort_values(by=['Total'],
                                                                                                 ascending=False)
    miss_data_columns = list(miss_data[miss_data['Percent'] >= miss_rate_filter].index)

    return df.drop(miss_data_columns, axis=1)


# 剔除常变量
# 原始数据中针对数值型特征，通过计算每个数值型特征的标准差，剔除部分变化很小的特征。这些变化很小的特征意味着区分度很低，可以直接清除掉。
def deal_change_small(df, filter_num):
    numberic_df = df.select_dtypes(include=[np.number]).std().sort_values()

    std_small_cols = list(numberic_df[numberic_df <= filter_num].index)
    return df.drop(std_small_cols, axis=1)


# 对某一列取对数
def data_col_log(df, col):
    df[col] = np.log1p(df[col])
    return df


# 标准化数据，一般针对是X
def scale_data(df):
    return preprocessing.scale(df)

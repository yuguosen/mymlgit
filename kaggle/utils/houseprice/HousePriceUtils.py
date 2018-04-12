# coding=utf-8
# 处理房价数据
import datetime
import pandas as pd
import numpy as np
import kaggle.utils.seabron.DealData as dd


# 特征的简单处理
def deal_data(df):
    df['MSSubClass'] = df['MSSubClass'].astype(str)
    now_year = datetime.datetime.now().year
    df['now_year'] = now_year

    # YearBuilt 这个是建造的年份转换成使用了的年份，及用当前的年份减去建造年份
    df['YearBuilt_Count'] = df['now_year'] - df['YearBuilt']
    # 删除YearBuilt
    df.drop(['YearBuilt'], axis=1)

    # YearRemodAdd  逻辑通YearBuilt
    df['YearRemodAdd_Count'] = df['now_year'] - df['YearRemodAdd']
    df.drop(['YearRemodAdd'], axis=1)
    df = df.drop(['now_year'], axis=1)

    # 非数值型进行编码
    df = pd.get_dummies(df)

    # 删除缺失数据较多的数据
    df = dd.deal_miss_rate(df, 0.2)
    # 删除防拆很小的数值型数据
    df = dd.deal_change_small(df, 0.4)
    # 用平均值填充缺失值
    df.fillna(df.mean())

    # 对y值取对数
    if 'SalePrice' in df.columns:
        df = dd.data_col_log(df, 'SalePrice')
    # 对GrLivArea取对数
    df = dd.data_col_log(df, 'GrLivArea')
    # 对TotalBsmtSF取对数
    df = dd.data_col_log(df, 'TotalBsmtSF')

    # 编码
    df = pd.get_dummies(df)

    return df


# 在对离散化变量进行编码的时候，test的离散值与训练集的离散值的数量不一样，需要补全
def deal_train_test(train_df, test_df):
    train_cols = list(train_df.columns)
    test_cols = list(test_df.columns)
    all_cols = list(set(train_cols).union(test_cols))
    for col in all_cols:
        if col not in train_df:
            train_df[col] = 0
        if col not in test_df:
            test_df[col] = 0

    return train_df, test_df


def get_data():
    train_data = deal_data(pd.read_csv(r'../house_price/data/train.csv'))
    test_data = deal_data(pd.read_csv(r'../house_price/data/test.csv'))

    y_train = train_data['SalePrice']
    x_train = train_data.drop(['SalePrice'], axis=1)

    x_train, x_test = deal_train_test(x_train, test_data)

    return x_train, y_train, x_test

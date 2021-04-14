import pandas as pd
import numpy as np


def clean_behavior_data(behaviors_source, behaviors_target):
    """
    提取需要的信息
    behaviors_source: 用户的历史数据
    +────────────────+───────────────────────────────────────────────────────────────────+
    | Column         | Content                                                           |
    +────────────────+───────────────────────────────────────────────────────────────────+
    | Impression ID  | 91                                                                |
    | User ID        | U397059                                                           |
    | Time           | 11/15/2019 10:22:32 AM                                            |
    | History        | N106403 N71977 N97080 N102132 N97212 N121652                      |
    | Impressions    | N129416-0 N26703-1 N120089-1 N53018-0 N89764-0 N91737-0 N29160-0  |
    +────────────────+───────────────────────────────────────────────────────────────────+
    处理后输出：
    +─────────────────+──────────────────────────────────────────────────────────+
    | Column          | Content                                                  |
    +─────────────────+──────────────────────────────────────────────────────────+
    | history         | N39144 N28742 N10369 N12912 N29465 N38587 N49827 N35943  |
    | candidate_news  | N11611                                                   |
    | clicked         | 0                                                        |
    +─────────────────+──────────────────────────────────────────────────────────+
    0表示没有点击，1表示点击了
    """
    behaviors = pd.read_table(behaviors_source, header=None, usecols=[3, 4], names=['history', 'impressions'])
    behaviors.impressions = behaviors.impressions.str.split()
    behaviors = behaviors.explode('impressions').reset_index(drop=True)
    behaviors['candidate_news'], behaviors['clicked'] = behaviors.impressions.str.split('-').str
    behaviors.history.fillna('', inplace=True)
    behaviors.to_csv(behaviors_target, sep='\t', index=False, columns=['history', 'candidate_news', 'clicked'])


if __name__ == '__main__':
    # clean_behavior_data("./data/train/behaviors.tsv", "./data/train/behaviors.csv")
    data = pd.read_csv("./data/train/behaviors.csv", sep='\t')
    print(data)

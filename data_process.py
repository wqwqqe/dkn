import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from config import Config


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


def clean_news_data(news_source, news_target):
    """
    news.tsv文件例子：
    +───────────────────+───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────+
    | Column            | Content                                                                                                                                   |
    +───────────────────+───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────+
    | News ID           | N37378                                                                                                                                    |
    | Category          | sports                                                                                                                                    |
    | SubCategory       | golf                                                                                                                                      |
    | Title             | PGA Tour winners                                                                                                                          |
    | Abstract          | A gallery of recent winners on the PGA Tour.                                                                                              |
    | URL               | https://www.msn.com/en-us/sports/golf/pga-tour-winners/ss-AAjnQjj?ocid=chopendata                                                         |
    | Title Entities    | [{"Label": "PGA Tour", "Type": "O", "WikidataId": "Q910409", "Confidence": 1.0, "OccurrenceOffsets": [0], "SurfaceForms": ["PGA Tour"]}]  |
    | Abstract Entites  | [{"Label": "PGA Tour", "Type": "O", "WikidataId": "Q910409", "Confidence": 1.0, "OccurrenceOffsets": [35], "SurfaceForms": ["PGA Tour"]}] |
    +───────────────────+───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────+
    输出：
    +─────────────────+───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────+
    | Column          | Content                                                                                                                                   |
    +─────────────────+───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────+
    | id              | N37378                                                                                                                                    |
    | title           | PGA Tour winners                                                                                                                          |
    | title entities  | [{"Label": "PGA Tour", "Type": "O", "WikidataId": "Q910409", "Confidence": 1.0, "OccurrenceOffsets": [0], "SurfaceForms": ["PGA Tour"]}]  |
    +─────────────────+───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────+

    """
    news = pd.read_table(news_source, header=None, usecols=[0, 3, 6], names=['id', 'title', 'entities'])
    news.to_csv(news_target, sep='\t', index=False)


def balance(source, target, true_false_rate):
    """
    对训练集的正样本和负样本的比例进行调整，使得其比例落在我们想要的区域内
    """
    low = true_false_rate[0]
    high = true_false_rate[1]
    assert low <= high
    behavior = pd.read_csv(source, sep='\t')
    true_part = behavior[behavior['clicked'] == 1]
    false_part = behavior[behavior['clicked'] == 0]
    if len(true_part) / len(false_part) < low:
        false_part = false_part.sample(n=int(len(true_part) / low))
    elif len(true_part) / len(false_part) > high:
        true_part = true_part.sample(n=int(len(false_part) * high))
    new_behavior = pd.concat([true_part, false_part]).sample(frac=1).reset_index(drop=True)
    new_behavior.to_csv(target, sep='\t', index=False)


def process_news_data(source, target, word2int_path, entity2int_path, mode):
    """
    将news_clean进一步处理
    +─────────────────+───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────+
    | Column          | Content                                                                                                                                   |
    +─────────────────+───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────+
    | id              | N37378                                                                                                                                    |
    | title           | PGA Tour winners                                                                                                                          |
    | title entities  | [{"Label": "PGA Tour", "Type": "O", "WikidataId": "Q910409", "Confidence": 1.0, "OccurrenceOffsets": [0], "SurfaceForms": ["PGA Tour"]}]  |
    +─────────────────+───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────+
    输出：
    +─────────────────+────────────────────────+
    | Column          | Content                |
    +─────────────────+────────────────────────+
    | id              | N37378                 |
    | title           | [1,2,3,0,0,0,0,0,0,0]  |
    | title entities  | [1,1,0,0,0,0,0,0,0,0]  |
    +─────────────────+────────────────────────+


    """

    def clean_text(text):
        return text.lower().strip()

    def process(news, word2int, entity2int):
        processed_news = []
        print(len(processed_news))
        with tqdm(total=len(news), desc="process words and entities") as qbar:
            for row in news.itertuples(index=False):
                new_row = [row.id, [0] * Config.num_words_a_news, [0] * Config.num_words_a_news]
                local_entity_map = {}
                for e in json.loads(row.entities):
                    if e['Confidence'] > Config.entity_confidence_threshold and e['WikidataId'] in entity2int:
                        for x in "".join(e['SurfaceForms']).lower().split():
                            local_entity_map[x] = entity2int[e['WikidataId']]

                try:
                    for i, w in enumerate(clean_text(row.title).split()):
                        if w in word2int:
                            new_row[1][i] = word2int[w]
                            if w in local_entity_map:
                                new_row[2][i] = local_entity_map[w]
                except IndexError:
                    pass

                processed_news.append(new_row)

                qbar.update(1)
        processed_news = pd.DataFrame(processed_news, columns=['id', 'title', 'entities'])

        return processed_news

    if mode == 'train':
        word2int = {}
        word2freq = {}
        entity2int = {}
        entity2freq = {}

        news = pd.read_csv(source, sep='\t')
        news.entities.fillna('[]', inplace=True)

        with tqdm(total=len(news), desc="Counting words and entities") as qbar:
            for row in news.itertuples(index=False):
                for w in clean_text(row.title).split():  # 统计标题中单词出现的频率
                    word2freq[w] = word2freq.get(w, 0) + 1
                for e in json.loads(row.entities):  # 统计实体出现的频率，按照置信度比例来算
                    times = len(list(filter(lambda x: x < len(row.title), e['OccurrenceOffsets']))) * e['Confidence']
                    if times > 0:
                        entity2freq[e['WikidataId']] = entity2freq.get(e['WikidataId'], 0) + times
                qbar.update(1)

        for key, value in word2freq.items():
            if value >= Config.word_freq_threshold:
                word2int[key] = len(word2int) + 1

        for key, value in entity2freq.items():
            if value >= Config.entity_freq_threshold:
                entity2int[key] = len(entity2int) + 1

        pd.DataFrame(word2int.items(), columns=['word', 'int']).to_csv(word2int_path, sep='\t', index=False)
        pd.DataFrame(entity2int.items(), columns=['entity', 'int']).to_csv(entity2int_path, sep='\t', index=False)

    elif mode == 'test':
        news = pd.read_table(source)
        news.entities.fillna('[]', inplace=True)

        word2int = dict(pd.read_csv(word2int_path, sep='\t').values.tolist())
        entity2int = dict(pd.read_csv(entity2int_path, sep='\t').values.tolist())
    else:
        print("wrong")
        return 0

    process(news, word2int, entity2int).to_csv(target, sep='\t', index=False)


def transform_entity_embedding(source, target, entity2int_path):
    """
    将.vec转换为.npy
    +────────────+────────────────────────────────────────────+
    | ID         | Embedding Values                           |
    +────────────+────────────────────────────────────────────+
    | Q42306013  | 0.014516	-0.106958 0.024590...	-0.080382 |
    +────────────+────────────────────────────────────────────+
    """
    entity_embedding = pd.read_table(source, header=None)
    entity_embedding['vector'] = entity_embedding.iloc[:, 1:101].values.tolist()

    entity_embedding = entity_embedding[[0, 'vector']].rename(columns={0: "entity"})

    entity2int = pd.read_csv(entity2int_path, sep='\t')
    merged_df = pd.merge(entity_embedding, entity2int, on='entity').sort_values('int')

    entity_embedding_transformed = np.zeros((len(entity2int) + 1, Config.entity_embedding_dim))
    for row in merged_df.itertuples(index=False):
        entity_embedding_transformed[row.int] = row.vector
    np.save(target, entity_embedding_transformed)
    print(len(entity_embedding_transformed))


if __name__ == '__main__':
    # clean_behavior_data("./data/train/behaviors.tsv", "./data/train/behaviors.csv")
    # clean_news_data("./data/train/news.tsv", "./data/train/news_clean.csv")
    # balance("./data/train/behaviors_clean.csv", "./data/train/behaviors_balance.csv", [0, 5, 1])
    process_news_data("./data/train/news_clean.csv", "./data/train/news_with_entity.csv", "./data/train/word2int.csv",
                      "./data/train/entity2int.csv", mode='train')
    transform_entity_embedding("./data/train/entity_embedding.vec", "./data/train/entity_embedding.npy",
                               "./data/train/entity2int.csv")

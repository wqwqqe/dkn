from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval
from config import Config


class DKNDataset(Dataset):
    def __init__(self, behaviors_path, news_with_entity_path):
        """
        behaviors:
        +─────────────────+──────────────────────────────────────────────────────────+
        | Column          | Content                                                  |
        +─────────────────+──────────────────────────────────────────────────────────+
        | history         | N39144 N28742 N10369 N12912 N29465 N38587 N49827 N35943  |
        | candidate_news  | N11611                                                   |
        | clicked         | 0                                                        |
        +─────────────────+──────────────────────────────────────────────────────────+
        news_with_entity:
        +─────────────────+────────────────────────+
        | Column          | Content                |
        +─────────────────+────────────────────────+
        | id              | N37378                 |
        | title           | [1,2,3,0,0,0,0,0,0,0]  |
        | entities        | [1,1,0,0,0,0,0,0,0,0]  |
        +─────────────────+────────────────────────+
        """
        super(Dataset, self).__init__()
        self.behaviors = pd.read_csv(behaviors_path, sep='\t')
        self.behaviors.history.fillna("", inplace=True)
        self.news_with_entity = pd.read_csv(news_with_entity_path, sep='\t', index_col='id',
                                            converters={'title': literal_eval, 'entities': literal_eval})

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        """
        {
            clicked:0,
            candidate_news:
                {
                    "word":[0]*num_words_a_news,
                    "entity":[0] * num_words_a_news
                }
            history:
                [
                    {
                        "word": [0] * num_words_a_news,
                        "entity": [0] * num_words_a_news
                    } * num_clicked_news_a_user
                ]
        }
        """

        def news2dict(news, df):
            if news in df.index:
                return {"word": df.loc[news].title, "entity": df.loc[news].entities}
            else:
                return {"word": [0] * Config.num_words_a_news, "entity": [0] * Config.num_words_a_news}

        item = {}
        row = self.behaviors.iloc[idx]
        item['clicked'] = row.clicked
        item['candidate_news'] = news2dict(
            row.candidate_news, self.news_with_entity)
        item['history'] = [news2dict(x, self.news_with_entity) for x in
                           row.history.split()[:Config.num_clicked_news_a_user]]

        padding = {
            "word": [0] * Config.num_words_a_news,
            "entity": [0] * Config.num_words_a_news
        }
        repeat_times = Config.num_clicked_news_a_user - len(item["history"])
        if repeat_times >= 0:
            item['history'].extend([padding] * repeat_times)

        return item


if __name__ == '__main__':
    data = DKNDataset("data/train/behaviors_balance.csv",
                      "data/train/news_with_entity.csv")
    print(data.behaviors)

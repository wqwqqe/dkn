import os


class Config():
    batch_size = 256
    num_workers = 4
    learning_rate = 0.001
    word_freq_threshold = 3
    entity_freq_threshold = 3
    num_epoch = 100
    num_words_a_news = 20  # 一篇资讯标题的单词数量
    entity_confidence_threshold = 0.5
    entity_embedding_dim = 100

    num_clicked_news_a_user = 50
    word_embedding_dim = 100
    entity_embedding_dim = 100
    num_word_tokens = 17720 + 1  # 根据word2int来

    window_sizes = [2, 3, 4]

    num_filters = 50

    use_context = False
    use_attention = os.environ['ATTENTION'] == '1' if "ATTENTION" in os.environ else True

    train_validation_split = (0.8, 0.2)

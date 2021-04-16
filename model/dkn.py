import torch
import torch.nn as nn
from model.kcnn import KCNN
from model.attention import Attention


class DKN(nn.Module):
    def __init__(self, config, entity_embedding, context_embedding):
        super(DKN, self).__init__()
        self.config = config
        self.kcnn = KCNN(config, entity_embedding, context_embedding)
        if self.config.use_attention:
            self.attention = Attention(config)

        self.dnn = nn.Sequential(
            nn.Linear(
                len(self.config.window_sizes) * 2 * self.config.num_filters, 16
            ),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, candidate_news, history):
        candidate_news_vector = self.kcnn(candidate_news)
        history_vector = torch.stack([self.kcnn(x) for x in history])
        if self.config.use_attention:
            user_vector = self.attention(candidate_news_vector, history_vector)
        else:
            user_vector = history_vector.mean(dim=1)
        p = self.dnn(torch.cat((user_vector, candidate_news_vector), dim=1)).squeeze(dim=1)
        return p

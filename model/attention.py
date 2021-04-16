import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.config = config
        self.dnn = nn.Sequential(
            nn.Linear(
                len(self.config.window_sizes) * 2 * self.config.num_filters, 16),
            nn.Linear(16, 1)
        )

    def forward(self, candidate_news_vector, history):
        candidate_expanded = candidate_news_vector.expand(
            self.config.num_clicked_news_a_user, -1, -1)
        clicked_news_weights = F.softmax(
            self.dnn(
                torch.cat((history, candidate_expanded), dim=-1)
            ).squeeze(-1).transpose(0, 1), dim=1
        )
        user_vector = torch.bmm(clicked_news_weights.unsqueeze(1),
                                history.transpose(0, 1)).squeeze(1)
        return user_vector

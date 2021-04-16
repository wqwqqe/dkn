import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class KCNN(nn.Module):
    def __init__(self, config, entity_embediing, context_embedding):
        super(KCNN, self).__init__()
        self.config = config
        self.word_embedding = nn.Embedding(config.num_word_tokens, config.word_embedding_dim)
        self.entity_embedding = entity_embediing
        self.context_embedding = context_embedding
        self.transform_matrix = nn.Parameter(
            torch.empty(self.config.word_embedding_dim, self.config.entity_embedding_dim))
        self.transform_bias = nn.Parameter(torch.empty(self.config.word_embedding_dim))

        self.conv_filters = nn.ModuleDict({
            str(x): nn.Conv2d(3 if self.config.use_context else 2, self.config.num_filters,
                              (x, self.config.word_embedding_dim))
            for x in self.config.window_sizes
        }

        )

        self.transform_matrix.data.uniform_(-0.1, 0.1)
        self.transform_bias.data.uniform_(-0.1, 0.1)

    def forward(self, news):
        """

        :param news:
                {
                "word": [Tensor(batch_size) * num_words_a_news],
                "entity":[Tensor(batch_size) * num_words_a_news]
                }
        return :
                vector:batch_size,len(window_size)*num_filters
        """
        word_vector = self.word_embedding(torch.stack(news["word"], dim=1).to(device))

        entity_vector = F.embedding(
            torch.stack(news["entity"], dim=1),
            torch.from_numpy(self.entity_embedding)).float().to(device)

        if self.config.use_context:
            context_vector = F.embedding(
                torch.stack(news["entity"], dim=1),
                torch.from_numpy(self.context_embedding)).float().to(device)

        b = self.config.batch_size
        n = self.config.num_words_a_news
        d = self.config.word_embedding_dim
        k = self.config.entity_embedding_dim

        transformed_entity_vector = torch.tanh(
            torch.add(
                torch.bmm(self.transform_matrix.expand(b * n, -1, -1),
                          entity_vector.view(b * n, k, 1)).view(b, n, d),
                self.transform_bias.expand(b, n, - 1)
            )
        )

        if self.config.use_context:
            transformed_context_vector = torch.tanh(
                torch.add(
                    torch.bmm(self.transform_matrix.expand(b * n, -1, -1),
                              context_vector.view(b * n, k, 1)).view(b, n, d),
                    self.transform_bias.expand(b, n, -1)
                )
            )

        if self.config.use_context:
            multi_channel_vector = torch.stack([
                word_vector, transformed_entity_vector, transformed_context_vector
            ], dim=1)

        else:
            multi_channel_vector = torch.stack([
                word_vector, transformed_entity_vector
            ], dim=1)

        pooled_vector = []
        for x in self.config.window_sizes:
            convoluted = self.conv_filters[str(x)](multi_channel_vector).squeeze(dim=3)
            activated = F.relu(convoluted)
            pooled = activated.max(dim=-1)[0]
            pooled_vector.append(pooled)

        final_vector = torch.cat(pooled_vector, dim=1)
        return final_vector

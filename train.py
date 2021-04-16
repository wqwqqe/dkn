from model.dkn import DKN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from config import Config
from dataset import DKNDataset
import numpy as np
import os
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train():
    dataset = DKNDataset('data/train/behaviors_balance.csv', "data/train/news_with_entity.csv")
    train_size = int(Config.train_validation_split[0] / sum(Config.train_validation_split) * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, (train_size, validation_size))
    train_dataloader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True,
                                  num_workers=Config.num_workers,
                                  drop_last=True)

    entity_embedding = np.load("./data/train/entity_embedding.npy")
    dkn = DKN(Config, entity_embedding, None).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(dkn.parameters(), lr=Config.learning_rate)
    loss_full = []
    epoch = 0

    with tqdm(total=Config.num_epoch, desc="Training") as qbar:
        for x in train_dataloader:
            y_pred = dkn(x['candidate_news'], x['history'])
            y = x["clicked"].float().to(device)
            loss = criterion(y_pred, y)
            loss_full.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            qbar.update(1)
            print(loss)

if __name__ == '__main__':
    train()

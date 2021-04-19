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


def latest_checkpoint(directory):
    if not os.path.exists(directory):
        return None
    all_checkpoints = {
        int(x.split('.')[-2].split('-')[-1]): x
        for x in os.listdir(directory)
    }
    if not all_checkpoints:
        return None
    return os.path.join(directory,
                        all_checkpoints[max(all_checkpoints.keys())])


def train():
    dataset = DKNDataset('data/train/behaviors_balance.csv',
                         "data/train/news_with_entity.csv")
    train_size = int(
        Config.train_validation_split[0] / sum(Config.train_validation_split) * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, (train_size, validation_size))
    train_dataloader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True,
                                  num_workers=Config.num_workers, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=True,
                                num_workers=Config.num_workers, drop_last=True)
    entity_embedding = np.load("./data/train/entity_embedding.npy")
    dkn = DKN(Config, entity_embedding, None).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(dkn.parameters(), lr=Config.learning_rate)
    loss_full = []
    epoch = 0
    num = 0
    if Config.load_checkpoint:
        checkpoint_path = latest_checkpoint("./result")
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path)
            dkn.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            dkn.train()

    with tqdm(total=(Config.num_epoch*(int(len(train_dataset)/Config.batch_size)+1)), desc="Training") as qbar:
        while epoch < (Config.num_epoch*(int(len(train_dataset)/Config.batch_size)+1)):
            for x in train_dataloader:
                y_pred = dkn(x['candidate_news'], x['history'])
                y = x["clicked"].float().to(device)
                loss = criterion(y_pred, y)
                loss_full.append(loss.item() * len(x))
                num += len(x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epoch % Config.num_batches_batch_loss == 0:
                    tqdm.write(
                        "epoch {},current loss {:6f} average loss:{:.6f}".format(epoch, loss, np.sum(loss_full)/num))

                if epoch % Config.num_batches_val_loss_acc == 0:
                    with torch.no_grad():
                        a, b, c, d, e = validate(dkn, val_dataloader)
                        tqdm.write("after %d epoch,loss is %f,acc is %f ,precision is %f,recall is %f,f1 is %f" %
                                   (epoch, a, b, c, d, e))

                if epoch % Config.num_batches_save_checkpoint == 0:
                    torch.save(
                        {
                            'model_state_dict': dkn.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch
                        }, "./result/ckpt-{}.pth".format(epoch)
                    )

                epoch += 1
                qbar.update(1)

    torch.save(
        {
            'model_state_dict': dkn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }, "./checkpoint/ckpt-{}.pth".format(epoch))


def validate(model, dataloader):
    criterion = nn.BCELoss()
    loss_full = []
    acc_full = []
    tp = 0
    tp_fp = 0
    tp_fn = 0
    cnt = 0
    for x in dataloader:
        print("hello"+str(cnt))
        cnt += 1
        y_pred = model(x['candidate_news'], x['history'])
        y = x["clicked"].float().to(device)
        loss = criterion(y_pred, y)
        loss_full.append(loss.item())
        y_predict = np.array([0 if i < 0.5 else 1 for i in y_pred])
        acc = np.sum(y_predict == y.cpu().numpy()) / Config.batch_size
        acc_full.append(acc)
        tp_fp += np.sum(y_predict == 1)
        tp_fn += np.sum(y.cpu().numpy() == 1)
        tp += np.sum(y.cpu().numpy()[y_predict == 1] == 1)
    precision = tp / tp_fp
    recall = tp / tp_fn
    f1 = 2*precision*recall/(precision+recall)
    return np.mean(loss_full), np.mean(acc_full), precision, recall, f1


def test():
    dataset = DKNDataset('data/test/behaviors_balance.csv',
                         "data/test/news_with_entity.csv")
    dataloader = DataLoader(
        dataset, batch_size=Config.batch_size, drop_last=True)
    entity_embedding = np.load("./data/test/entity_embedding.npy")
    dkn = DKN(Config, entity_embedding, None).to(device)
    checkpoint_path = latest_checkpoint("./result")
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        dkn.load_state_dict(checkpoint['model_state_dict'])
        dkn.train()
    with torch.no_grad():
        a, b, c, d, e = validate(dkn, dataloader)
        print("loss is %f,acc is %f ,precision is %f,recall is %f,f1 is %f" %
              (a, b, c, d, e))


if __name__ == '__main__':
    test()

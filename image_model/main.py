import os
import re
import time
import glob
import json
import math
import random
import ytreader

import torch
import torch.nn.functional as F
import torchvision.models as models

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import cv2
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from dataset import get_dataloader
from model import Net
from evaluate_overlap import average_aggregated_score

from utils import DotDict
from config import config

from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader


try:
    import nirvana_dl
except ImportError:
    print("No nirvana_dl available")
    nirvana_dl = None
except Exception as e:
    raise e


seed = 19
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

class image_classification_dataset(Dataset):
    def __init__(self, file_names, labels, root_dir, transform=None, pad_square=False):
        self.labels = labels
        self.file_names = file_names
        self.root_dir = root_dir
        self.transform = transform
        self.pad_square = pad_square
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.file_names[idx])
        image = Image.open(img_name).convert('RGB')
        if self.pad_square:
            x, y = image.size
            maxside = max(x, y)
            delta_w = maxside - x
            delta_h = maxside - y
            padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
            image = ImageOps.expand(image, padding)

        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]

        return image, label


def last_checkpoint(output):
    # def corrupted(fpath):
    #     try:
    #         torch.load(fpath, map_location="cpu")
    #         return False
    #     except Exception as e:
    #         print(f"WARNING: Cannot load {fpath}")
    #         return True

    saved = sorted(
        glob.glob(f"{output}/checkpoint_*.pt"),
        key=lambda f: int(re.search("_(\d+).pt", f).group(1)),
    )

    if len(saved) > 0:
        last = saved[-1]

        for ckpt_path in saved[:-1]:
            os.remove(ckpt_path)

        return last

    # if len(saved) >= 1: # and not corrupted(saved[-1]):
    # return saved[-1]
    # elif len(saved) >= 2:
    # return saved[-2]
    # else:
    # return None


def best_checkpoint(output):
    saved = sorted(
        glob.glob(f"{output}/best_val_checkpoint_*.pt"),
        key=lambda f: int(re.search("_(\d+).pt", f).group(1)),
    )

    if len(saved) > 0:
        return saved[-1]


def save_checkpoint(model, optimizer, scheduler, epoch, filepath):
    print(f"Saving model and optimizer state at epoch {epoch} to {filepath}")
    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }

    # all_checkpoints = glob.glob(os.path.join(os.path.dirname(filepath), "checkpoint_*.pt"))
    # all_checkpoints = sorted(all_checkpoints, key=lambda x: os.path.getctime(x))
    torch.save(checkpoint, filepath)
    try:
        import nirvana_dl

        nirvana_dl.snapshot.dump_snapshot()
    except ImportError:
        print("Can not import nirvana_dl. Continue without dumping nirvana snapshot")
    except Exception as e:
        raise e


def remove_old_checkpoints(current_epoch, dir_path, file_starting_name="checkpoint"):
    for i in range(current_epoch):
        file_path = os.path.join(dir_path, f"{file_starting_name}_{i}.pt")
        if os.path.isfile(file_path):
            os.remove(file_path)


def load_checkpoint(model, optimizer, scheduler, filepath):
    print(f"Loading model and optimizer state from {filepath}")
    checkpoint = torch.load(filepath, map_location="cpu")
    epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    return epoch


def train(config):
    model = Net(config.model_config).to(config.device)
    print(model)
    print([k for k,_ in model.named_parameters()])
    label2idx = {w: idx for idx, w in enumerate(config.dataset.class_names)}
    idx2label = {idx: w for w, idx in label2idx.items()}

    hp = pd.read_csv(config.honeypots_filename)
    classes = [label2idx[x] for x in np.array(hp['label'])]
    hp["class"] = classes

    print("Tolokers average aggregated score: ", average_aggregated_score(hp, group_by_col='input_image', overlap=config.pool_overlap))

    train_loader = get_dataloader(
        config.train_table_path,
        transform=transforms.Compose(
            [
                transforms.Resize(config.dataset.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        batch_size=config.batch_size,
        class_mapping=label2idx
    )

    val_loader = get_dataloader(
        config.val_table_path,
        transform=transforms.Compose(
            [
                transforms.Resize(config.dataset.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        class_mapping=label2idx
    )

    if config.experiment.loss == "NLL":
        loss_function = F.nll_loss
    else:
        raise Exception("unknown loss")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.experiment.lr,
        weight_decay=config.experiment.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, verbose=True, patience=config.experiment.sched_patience, factor=config.experiment.sched_factor
    )

    os.makedirs(config.results_dir, exist_ok=True)

    if nirvana_dl is not None:
        log_dir = nirvana_dl.logs_path()
    else:
        log_dir = config.results_dir
    writer = SummaryWriter(log_dir=log_dir)

    ckpt = last_checkpoint(config.results_dir)
    if ckpt is None:
        epoch = 0
    else:
        epoch = load_checkpoint(model, optimizer, scheduler, ckpt)

    best_val = 1e9
    train_loss = None

    while epoch < config.experiment.n_epochs:
        print("==== training ====")
        model.train()
        torch.cuda.empty_cache()

        preds = []
        true = []
        probs = []

        for i, (inputs, labels) in enumerate(train_loader):
            if i % 100 == 0:
                print(i)
            
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()

            optimizer.step()
            loss = loss.detach().cpu().numpy()
            train_loss = (
                train_loss * 0.2 + loss * 0.8 if train_loss is not None else loss
            )

            pred = torch.argmax(outputs, 1)
            preds.append(pred.detach().to("cpu").clone())
            true.append(labels.detach().to("cpu").clone())
            probs.append(outputs.detach().to("cpu").clone())

        preds = torch.cat(preds).numpy()
        true = torch.cat(true).numpy()
        probs = torch.cat(probs).numpy()

        train_scores = {
            "loss": train_loss,
            "accuracy": accuracy_score(true, preds),
            "f1_micro": f1_score(true, preds, average="micro"),
            "f1_macro": f1_score(true, preds, average="macro")
        }
        print(train_scores)
        writer.add_scalar("train loss", train_scores["loss"], epoch)

        print("==== validation ====")

        model.eval()
        torch.cuda.empty_cache()
        total_val_loss = 0

        with torch.no_grad():
            preds = torch.tensor([])
            true = torch.tensor([])
            probs = torch.tensor([])
            for i, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(config.device)
                labels = labels.to(config.device)
                outputs = model(inputs)

                pred = torch.argmax(outputs, 1)
                preds = torch.cat([preds, pred.to("cpu")])
                true = torch.cat([true, labels.to("cpu")])
                probs = torch.cat([probs, outputs.to("cpu")])

                val_loss = loss_function(outputs, labels)

                total_val_loss = (
                    total_val_loss * i + val_loss.cpu().numpy().item()
                ) / (i + 1)

            preds = np.array(preds)
            true = np.array(true)
            probs = np.array(torch.exp(probs))

            scheduler.step(total_val_loss)

            if best_val > total_val_loss:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    os.path.join(config.results_dir, f"best_val_checkpoint_{epoch}.pt"),
                )
                best_val = total_val_loss

                remove_old_checkpoints(epoch, config.results_dir)

            scores = {
                "loss": val_loss,
                "accuracy": accuracy_score(true, preds),
                "f1_micro": f1_score(true, preds, average="micro"),
                "f1_macro": f1_score(true, preds, average="macro"),
            }

            print("val scores", scores)

            writer.add_scalar("train loss", train_scores["loss"], epoch)
            writer.add_scalar("val loss", scores["loss"], epoch)
            writer.add_scalar("accuracy", scores["accuracy"], epoch)
            writer.add_scalar("f1 micro", scores["f1_micro"], epoch)
            writer.add_scalar("f1 macro", scores["f1_macro"], epoch)
            # writer.add_scalar("ROC-AUC", scores["rocauc"], epoch)

            # writer.add_scalar("test accuracy", tacc)
            # writer.add_scalar("test average aggregated accuracy", taasm)

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                os.path.join(config.results_dir, f"checkpoint_{epoch}.pt"),
            )

        test_model(config, model, epoch, hp, label2idx)
        epoch += 1


def test_model(config, model, epoch_num, hp, label2idx):
    print("==== test ====")

    idx2label = {idx: w for w, idx in label2idx.items()}

    hp_agg = {x[0]:x[1] for x in np.array(hp[['input_image', 'class']])}
    hp_agg = pd.DataFrame(data={'input_image': hp_agg.keys(), 'class': hp_agg.values()})

    test_loader = get_dataloader(
        config.test_table_path,
        transform=transforms.Compose(
            [
                transforms.Resize(config.dataset.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        batch_size=1,
        class_mapping=label2idx,
        mode="test"
    )

    model.eval()

    with torch.no_grad():
        preds = []
        true = []
        probs = []
        img_names = []
        for i, (img_key, inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)
            outputs = model(inputs)

            pred = torch.argmax(outputs, 1)
            preds.append(pred.detach().to("cpu").clone())
            true.append(labels.detach().to("cpu").clone())
            probs.append(outputs.detach().to("cpu").clone())

            img_names.append(img_key)

        preds = torch.cat(preds).numpy()
        true = torch.cat(true).numpy()
        probs = torch.cat(probs).numpy()
        img_names = np.concatenate(img_names, axis=0)

        test_df = pd.DataFrame({"input_image": img_names, "class": preds})
        test_df["label"] = test_df["class"].map(idx2label)
        hp = hp[hp["input_image"].isin(img_names)]

        test_df.to_csv("test_df_my.csv", index=False)
        hp.to_csv("hp_my.csv", index=False)

        taasm, _ = average_aggregated_score(hp, test_df, group_by_col='input_image', n_resamples=50, overlap=config.pool_overlap)
        print("av ag score", taasm)


    scores = {
        "epoch": epoch_num,
        "accuracy": accuracy_score(true, preds),
        "f1_micro": f1_score(true, preds, average="micro"),
        "f1_macro": f1_score(true, preds, average="macro"),
        "average_aggregated_score": taasm
    }

    print("test scores", scores)

    if nirvana_dl is not None:
        json_path = os.path.join(nirvana_dl.snapshot.get_snapshot_path(), f"test_metrics_{epoch_num}.json")
        with open(json_path, "w") as out:
            json.dump(scores, out)


if __name__ == "__main__":
    if nirvana_dl is not None:
        config = nirvana_dl.params()
        config["results_dir"] = os.path.join(
            nirvana_dl.snapshot.get_snapshot_path(), config["results_dir"]
        )
        config["model_config"]["body_weights_file"] = os.path.join(
            nirvana_dl.snapshot.get_snapshot_path(), config["model_config"]["body_weights_file"]
        )
        config["honeypots_filename"] = os.path.join(
            nirvana_dl.snapshot.get_snapshot_path(), config["honeypots_filename"]
        )

    config = DotDict(config)

    train(config)
    
    if nirvana_dl is not None:
        result_json = {}
        test_output_files = glob.glob(os.path.join(nirvana_dl.snapshot.get_snapshot_path(), "test_metrics_*.json"))
        best_score = 0
        for json_path in test_output_files:
            with open(json_path) as json_file:
                data = json.load(json_file)
                if data["average_aggregated_score"] > best_score:
                    best_score = data["average_aggregated_score"]
                    result_json["best_checkpoint"] = data

        result_json["config"] = config
        with open(nirvana_dl.json_output_file(), "w") as out:
            json.dump(result_json, out)
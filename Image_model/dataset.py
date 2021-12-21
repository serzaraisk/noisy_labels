import os
import math
import cv2
import numpy as np

import ytreader

from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader

import torch
import torch.utils.data as torch_data

from PIL import Image, ImageOps


class DataProcessor:
    def __init__(self, transform=None, class_mapping=None, mode="train"):
        self.transform = transform
        self.class_mapping = class_mapping
        self.mode = mode
        # self.pad_square = None

    def __call__(self, row):
        row = {k.decode("utf-8"): v for k, v in row.items()}
        nparr = np.frombuffer(row["value"], np.uint8)
        if len(nparr) == 0:
            return None
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_np is None:
            return None

        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img_np).convert("RGB")

        x, y = image.size
        maxside = max(x, y)
        delta_w = maxside - x
        delta_h = maxside - y
        padding = (
            delta_w // 2,
            delta_h // 2,
            delta_w - (delta_w // 2),
            delta_h - (delta_h // 2),
        )
        image = ImageOps.expand(image, padding)

        if self.transform is not None:
            image = self.transform(image)

        label = row["label"].decode("utf-8")
        if self.class_mapping is not None:
            label = self.class_mapping[label]
        else:
            label = int(label)

        label = torch.tensor(label)

        if self.mode == "test":
            return row["key"].decode("utf-8"), image, label

        return image, label


class IterableYTDataset(IterableDataset):
    def __init__(self, table, row_processor, num_readers=4):
        self._table = table
        self._row_processor = row_processor
        self._num_readers = num_readers
        self._reader = None

    def __iter__(self):
        local_num_workers, local_worker_idx = (
            get_local_num_workers(),
            get_local_worker_id(),
        )
        num_workers = local_num_workers
        worker_idx = local_worker_idx
        if self._reader is None:
            local_num_workers = get_local_num_workers()
            self._reader = _make_reader(
                self._table,
                num_workers,
                worker_idx,
                yt_num_readers=max(1, self._num_readers // local_num_workers),
            )
        self._reader.reset_to_row(0)
        for row in self._reader:
            processed_row = self._row_processor(row)
            if processed_row is None:
                continue
            yield processed_row


def get_local_num_workers():
    num_local = 1
    local_worker_info = torch_data.get_worker_info()
    if local_worker_info is not None:
        num_local = local_worker_info.num_workers
    return num_local


def get_local_worker_id():
    id_local = 0
    local_worker_info = torch_data.get_worker_info()
    if local_worker_info is not None:
        id_local = local_worker_info.id
    return id_local


def _make_reader(
    table,
    num_workers,
    worker_idx,
    yt_num_readers=1,
    yt_cache_size=1024,
    yt_cluster="hahn",
):

    reader = ytreader.YTTableParallelReader(
        yt_cluster, table, yt_cache_size, yt_num_readers
    )
    if num_workers > 1:
        num_rows_per_worker = int(math.ceil(reader.num_rows / num_workers))
        first_row_idx = worker_idx * num_rows_per_worker
        last_row_idx = min(first_row_idx + num_rows_per_worker, reader.num_rows)
        return reader.make_subset_reader(first_row_idx, last_row_idx)
    return reader


def get_dataloader(table, transform=None, batch_size=256, class_mapping=None, mode="train", num_workers=8):
    row_processor = DataProcessor(transform=transform, class_mapping=class_mapping, mode=mode)
    dataset = IterableYTDataset(table, row_processor, num_readers=num_workers)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    return loader

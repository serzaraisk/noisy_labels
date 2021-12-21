import numpy as np

config = {
    "device": "cuda:0",
    "batch_size": 128,
    "train_table_path": "//home/toloka-analytics/loompa/watermark_model/train_with_imgs",
    "val_table_path": "//home/toloka-analytics/loompa/watermark_model/val_with_imgs",
    "test_table_path": "//home/toloka-analytics/loompa/watermark_model/test_with_imgs",
    "pool_overlap": 3,
    "honeypots_filename": "hp_watermarks.csv",
    "graph_config": {
        "batch_size": 10,
        "honeypots_nirvana_data": "4c4472d8-d8a4-429b-8168-8b14237bb8d2",
        "weights_nirvana_data": "12f0c103-97c1-4526-97d0-7943a2db08ba",
        "torch_layer": "0e70f058-e980-44ff-b54c-f9f8c17d200c",
    },
    "model_config": {
        "body_weights_file": "efficientnet_b2_weights",
        "model_body": "efficientNet",
        "n_classes": 2,
        "hidden_size": 256,
        "dropout": 0.4,
    },
    "dataset": {"image_size": 288, "class_names": ["clean", "has_watermark"]},
    "results_dir": "model_results",
    "experiment": {
        "loss": "NLL",
        "n_epochs": 10,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "sched_patience": 3,
        "sched_factor": 0.5,
    },
}

param_grid = {
    "loss": ["NLL"],
    "n_epochs": [10],
    "lr": [1e-4],
    "weight_decay": [0.01],
    # "dropout": [0.2, 0.5],
    "sched_patience": [3],
    "sched_factor": [0.5],
}
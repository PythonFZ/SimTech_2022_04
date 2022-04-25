import pathlib

import kaggle
from zntrack import NodeConfig, nodify


@nodify(
    outs=pathlib.Path("dataset"), params={"dataset": "datamunge/sign-language-mnist"}
)
def download_kaggle(cfg: NodeConfig):
    """Download dataset from kaggle"""
    kaggle.api.dataset_download_files(
        dataset=cfg.params.dataset, path=cfg.outs, unzip=True
    )

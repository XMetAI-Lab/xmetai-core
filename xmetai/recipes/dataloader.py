#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reusable LazyConfig recipe for Zarr-backed weather datasets."""

from typing import Optional, Sequence

from omegaconf import OmegaConf

from xmetai import DATASETS, METRICS, build_test_loader, build_train_loader
from xmetai.config import LazyCall as L


def zarr_dataloader(
    data_paths: Sequence[str],
    dataset_name: str = "MultiZarrDataset",
    dataset_kwargs: Optional[dict] = None,
    in_frames: int = 2,
    train_fcst_frames: int = 1,
    test_frames: Optional[Sequence[int]] = None,
    freq: int = 24,
    train_times: Optional[Sequence[str]] = None,
    test_times: Optional[Sequence[str]] = None,
    total_batch_size: int = 1,
    train_total_batch_size: Optional[int] = None,
    test_total_batch_size: Optional[int] = None,
    train_workers: int = 8,
    test_workers: int = 4,
    train_interval: int = 1,
    test_interval: int = -1,
    train_pin_memory: bool = True,
    test_pin_memory: bool = True,
    train_do_augment: bool = False,
    eval_test_names: Optional[Sequence[str]] = None,
    test_target_names: Optional[Sequence[str]] = None,
    in_names: Optional[Sequence[str]] = None,
    lat_flip: bool = False,
    start_frames: int = 0,
):
    in_names = [] if in_names is None else list(in_names)
    test_frames = list(range(1, 43)) if test_frames is None else list(test_frames)
    eval_test_names = [] if eval_test_names is None else list(eval_test_names)
    test_target_names = [] if test_target_names is None else list(test_target_names)
    dataset_kwargs = {} if dataset_kwargs is None else dict(dataset_kwargs)
    train_total_batch_size = total_batch_size if train_total_batch_size is None else train_total_batch_size
    test_total_batch_size = total_batch_size if test_total_batch_size is None else test_total_batch_size

    dataloader = OmegaConf.create()

    train_dataset_kwargs = dict(
        data_paths=data_paths,
        hist_frames=in_frames,
        fcst_frames=train_fcst_frames,
        start_frames=start_frames,
        interval=train_interval,
        freq=freq,
        training=True,
        years=train_times,
        in_names=in_names,
        lat_flip=lat_flip,
        **dataset_kwargs,
    )
    test_dataset_kwargs = dict(
        data_paths=data_paths,
        hist_frames=in_frames,
        fcst_frames=test_frames,
        start_frames=start_frames,
        interval=test_interval,
        freq=freq,
        training=False,
        inference_only=False,
        years=test_times,
        in_names=in_names,
        lat_flip=lat_flip,
        **dataset_kwargs,
    )
    if test_target_names:
        test_dataset_kwargs["target_names"] = test_target_names

    dataloader.train = L(build_train_loader)(
        dataset=L(DATASETS.get(dataset_name))(**train_dataset_kwargs),
        total_batch_size=train_total_batch_size,
        num_workers=train_workers,
        pin_memory=train_pin_memory,
        do_augment=train_do_augment,
    )

    dataloader.test = L(build_test_loader)(
        dataset=L(DATASETS.get(dataset_name))(**test_dataset_kwargs),
        total_batch_size=test_total_batch_size,
        num_workers=test_workers,
        pin_memory=test_pin_memory,
    )

    eval_times = [int(t * freq) for t in test_frames]
    dataloader.evaluator = [
        L(METRICS.get("MSE"))(
            test_names=eval_test_names,
            test_times=eval_times,
        ),
    ]

    return dataloader


MultiZarr = zarr_dataloader

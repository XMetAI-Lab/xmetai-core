import numpy as np
import torch
import torch.utils.data as torchdata

from xmetai.utils.comm import get_world_size
from xmetai.utils.env import seed_all_rng

from .sampler import InferenceSampler, TrainingSampler


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    data = {}
    for k in batch[0]:
        imgs = [x[k] for x in batch]

        if torch.is_tensor(imgs[0]):
            data[k] = torch.stack(imgs)
        else:
            data[k] = np.stack(imgs)
    return data


def augment_batch_collator(batch, max_roll=30):
    """
    A batch collator that applies data augmentation.
    Longitude shift is applied to the input and target.
    The shift is random for each sample.
    """
    batch = trivial_batch_collator(batch)
    # Apply data augmentation here
    if "inputs" in batch and len(batch["inputs"]) > 0:
        W = batch["inputs"].shape[-1]
        shift = torch.randint(0, min(W, max_roll), (batch["inputs"].shape[0],))
        # longitude shift
        batch["inputs"] = torch.stack([torch.roll(batch["inputs"][i], shifts=int(s), dims=-1) for i, s in enumerate(shift)])
        if "targets" in batch and len(batch["targets"]) > 0:
            batch["targets"] = torch.stack([torch.roll(batch["targets"][i], shifts=int(s), dims=-1) for i, s in enumerate(shift)])
        batch["shift"] = shift
    return batch


def worker_init_reset_seed(worker_id):
    initial_seed = torch.initial_seed() % 2**31
    seed_all_rng(initial_seed + worker_id)


def build_train_loader(
    dataset,
    *,
    sampler=None,
    total_batch_size,
    num_workers=0,
    pin_memory=True,
    do_augment=True,
):
    world_size = get_world_size()
    batch_size = total_batch_size // world_size

    if sampler is None:
        sampler = TrainingSampler(len(dataset))

    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=augment_batch_collator if do_augment else trivial_batch_collator,
        worker_init_fn=worker_init_reset_seed,
        persistent_workers=True,
        pin_memory=pin_memory,
        drop_last=True,
    )


def build_test_loader(
    dataset,
    *,
    sampler=None,
    total_batch_size,
    num_workers=0,
    pin_memory=True,
):
    world_size = get_world_size()
    batch_size = total_batch_size // world_size

    if sampler is None:
        sampler = InferenceSampler(len(dataset), drop_last=True)

    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        persistent_workers=True,
        pin_memory=pin_memory,
        drop_last=True,
    )

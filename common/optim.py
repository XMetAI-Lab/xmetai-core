import torch

from xmetai.config import LazyCall as L
from xmetai.core import get_default_optimizer_params, get_muon_params_group
from xmetai.core.muon import MuonWithAuxAdam

SGD = L(torch.optim.SGD)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        weight_decay_norm=0.0
    ),
    lr=0.02,
    momentum=0.9,
    weight_decay=1e-4,
)


AdamW = L(torch.optim.AdamW)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        base_lr="${..lr}",
        weight_decay_norm=0.0,
    ),
    lr=5e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)

Adam = L(torch.optim.Adam)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        base_lr="${..lr}",
        weight_decay_norm=0.0,
    ),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
)

Muon = L(MuonWithAuxAdam)(
    params=L(get_muon_params_group)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        base_lr="${..lr}",
    ),
    lr=5e-4,
)
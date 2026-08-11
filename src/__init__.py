"""Project package with eagerly registered model components."""

__all__ = ["Model"]

import torch.nn as nn

from .models import Model, module_type2class
from .modules.poolers import (
    MaxPooler, MeanPooler, MeanStartEndMaxPooler, MeanStartMaxPooler,
    MeanStdStartEndMaxMinPooler, NoAffinePooler, StartPooler,
)
from .modules.sequence import (
    AttentionDecoder, CrossEntropyLoss, GreedyDecoder, MaskMaker,
    PositionalEmbedding, SelfAttentionLayer, TeacherForcer,
    TransformerEncoder,
)
from .modules.tunnel import Layer, Tunnel
from .modules.vae import MinusD_KLLoss, VAE

classes = [
    nn.MSELoss, nn.BCEWithLogitsLoss, Layer, Tunnel,
    TeacherForcer, MaskMaker, SelfAttentionLayer, PositionalEmbedding,
    TransformerEncoder, AttentionDecoder, GreedyDecoder, CrossEntropyLoss,
    VAE, MinusD_KLLoss, MeanPooler, StartPooler, MaxPooler,
    MeanStartMaxPooler, MeanStartEndMaxPooler,
    MeanStdStartEndMaxMinPooler, NoAffinePooler,
]
for cls in classes:
    module_type2class[cls.__name__] = cls

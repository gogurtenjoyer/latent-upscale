import os
from typing import Literal

import torch
import torch.nn as nn
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from invokeai.app.invocations.fields import FieldDescriptions

from invokeai.invocation_api import (
    BaseInvocation,
    Input,
    InputField,
    InvocationContext,
    invocation,
    LatentsField,
    LatentsOutput
)


class Upscaler(nn.Module):
    """
        Basic NN layout, ported from:
        https://github.com/city96/SD-Latent-Upscaler/blob/main/upscaler.py
    """
    version = 2.1 # network revision
    def head(self):
        return [
            nn.Conv2d(self.chan, self.size, kernel_size=self.krn, padding=self.pad),
            nn.ReLU(),
            nn.Upsample(scale_factor=self.fac, mode="nearest"),
            nn.ReLU(),
        ]
    def core(self):
        layers = []
        for _ in range(self.depth):
            layers += [
                nn.Conv2d(self.size, self.size, kernel_size=self.krn, padding=self.pad),
                nn.ReLU(),
            ]
        return layers
    def tail(self):
        return [
            nn.Conv2d(self.size, self.chan, kernel_size=self.krn, padding=self.pad),
        ]

    def __init__(self, fac, depth=16):
        super().__init__()
        self.size = 64      # Conv2d size
        self.chan = 4       # in/out channels
        self.depth = depth  # no. of layers
        self.fac = fac      # scale factor
        self.krn = 3        # kernel size
        self.pad = 1        # padding

        self.sequential = nn.Sequential(
            *self.head(),
            *self.core(),
            *self.tail(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)


SD_VERSIONS = Literal[('v1', 'xl')]
FACTORS = Literal[('1.25', '1.5', '2.0')]


@invocation(
    "upscale_latents",
    title="Upscale Latents",
    tags=["latents", "upscale"],
    category="latents",
    version="1.1.0",
)
class UpscaleLatentsInvocation(BaseInvocation):
    """Upscales latents using a trained model"""

    latents: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    latent_ver: SD_VERSIONS = InputField(default='v1', input=Input.Direct)
    scale_factor: FACTORS = InputField(default='2.0', input=Input.Direct)
    magic_number: float = InputField(default=0.18215, description="Unless you have a great reason, do not change from 0.18215!")

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = context.tensors.load(self.latents.latents_name)

        latents = torch.div(latents, self.magic_number) # MAGIC NUMBER! YOU'LL SEE ME AGAIN!!

        model = Upscaler(self.scale_factor)

        filename = f"latent-upscaler-v{model.version}_SD{self.latent_ver}-x{self.scale_factor}.safetensors"
        local = os.path.join(
            os.path.join(os.path.dirname(os.path.realpath(__file__)),"models"),
            filename
        )

        if os.path.isfile(local):
            context.logger.info("[Upscale Latents] Using Local Model")
            weights = local
        else:
            context.logger.info("[Upscale Latents] Using HF Hub Model")
            weights = str(hf_hub_download(
                repo_id="city96/SD-Latent-Upscaler",
                filename=filename)
            )


        model.load_state_dict(load_file(weights))
        lt = latents.to("cpu", torch.float32)
        resized_latents = model(lt)
        resized_latents = torch.mul(resized_latents, self.magic_number) # HELLO AGAIN FROM YOUR OLD FRIEND, MAGIC NUMBER!
        del model

        name = context.tensors.save(tensor=resized_latents)
        return LatentsOutput.build(latents_name=name, latents=resized_latents, seed=self.latents.seed)


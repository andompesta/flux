from math import ceil

import torch

from flux.trt.engine.base_engine import BaseEngine
from flux.trt.mixin.t5_mixin import T5Mixin


class T5Engine(T5Mixin, BaseEngine):
    def __init__(
        self,
        text_maxlen: int,
        hidden_size: int,
        engine_path: str,
    ):
        super().__init__(
            text_maxlen=text_maxlen,
            hidden_size=hidden_size,
            engine_path=engine_path,
        )

    def __call__(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        assert latent.device == self.tensors["latent"].device, "device mismatch | expected {}; actual {}".format(
            self.tensors["latent"].device,
            latent.device,
        )

        assert latent.dtype == self.tensors["latent"].dtype, "dtype mismatch | expected {}; actual {}".format(
            self.tensors["latent"].dtype,
            latent.dtype,
        )

        feed_dict = {"latent": latent}
        images = self.infer(feed_dict=feed_dict)["images"].clone()
        return images

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.__call__(z)

    def get_shape_dict(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
    ) -> dict[str, tuple]:

        return {
            "input_ids": (batch_size, self.text_maxlen),
            "text_embeddings": (batch_size, self.text_maxlen, self.hidden_size),
        }

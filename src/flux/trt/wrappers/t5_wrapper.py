import torch
import onnx_graphsurgeon as gs

from flux.modules.conditioner import HFEmbedder
from .base_wrapper import BaseWrapper, Optimizer, TransformersModelWrapper


class T5Wrapper(BaseWrapper):
    def __init__(
        self,
        model: HFEmbedder,
        fp16=False,
        tf32=False,
        bf16=False,
        max_batch=16,
        verbose=True,
    ):
        exp_model = TransformersModelWrapper(model=model, output_name="last_hidden_state")
        super().__init__(
            model=exp_model,
            fp16=fp16,
            tf32=tf32,
            bf16=bf16,
            max_batch=max_batch,
            verbose=verbose,
        )

        # set proper dtype
        self.prepare_model()

    def get_input_names(self):
        return ["input_ids"]

    def get_output_names(self):
        output_names = ["text_embeddings"]
        return output_names

    def get_dynamic_axes(self):
        dynamic_axes = {
            "input_ids": {0: "B"},
            "text_embeddings": {0: "B"},
        }
        return dynamic_axes

    def check_dims(
        self,
        batch_size: int,
    ) -> None | tuple[int, int]:
        assert batch_size >= self.min_batch and batch_size <= self.max_batch

    def get_sample_input(
        self,
        batch_size: int,
        opt_image_height: int,
        opt_image_width: int,
    ) -> torch.Tensor:
        self.check_dims(batch_size)
        return torch.zeros(
            batch_size,
            self.model.text_maxlen,
            dtype=torch.int32,
            device=self.device,
        )

    def get_input_profile(
        self,
        batch_size: int,
        image_height: int,
        image_width: int,
    ):
        self.check_dims(batch_size)
        return {
            "input_ids": [
                (self.min_batch, self.model.text_maxlen),
                (batch_size, self.model.text_maxlen),
                (self.max_batch, self.model.text_maxlen),
            ]
        }

    def optimize(self, onnx_graph, return_onnx=True, *args, **kwargs):
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ": original")
        if kwargs.get("modify_fp8_graph", False):
            opt.modify_fp8_graph()
            opt.info(self.name + ": modify fp8 graph")
        else:
            opt.fold_constants()
            opt.info(self.name + ": fold constants")
            opt.infer_shapes()
            opt.info(self.name + ": shape inference")
            if kwargs.get("fuse_mha_qkv_int8", False):
                opt.fuse_mha_qkv_int8_sq()
                opt.info(self.name + ": fuse QKV nodes")

        onnx_opt_graph = gs.export_onnx(opt.graph)
        opt.info(self.name + ": finished")
        return onnx_opt_graph

import os
from functools import partial

import numpy as np
import onnx
import onnxruntime
import torch

from flux.util import load_ae, load_clip, load_flow_model, load_t5

from src.flux.trt.onnx_export import CLIPExporter, TransformerExporter, T5Exporter, VAEExporter

MODEL_CONFIG = {
    "vae": {
        "wrapper": VAEExporter,
        "loader": partial(load_ae, name="flux-dev"),
    },
    "clip": {
        "wrapper": lambda x: CLIPExporter(x, fp16=True),
        "loader": lambda device: load_clip(device),
    },
    "t5": {
        "wrapper": T5Exporter,
        "loader": lambda device: load_t5(device, max_length=512),
    },
    "transformer": {
        "wrapper": lambda x: TransformerExporter(x, fp16=True),
        "loader": lambda device: load_flow_model(device=device, name="flux-dev"),
    },
}


def to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
    device = torch.device("cuda")
    batch_size = 1
    width: int = 1360
    height: int = 768

    base_path = "/workspace/data/flux/dd-onnx-bf16"
    models_name = [
        "vae",
        # "clip",
        # "t5",
        # "transformer",
    ]

    for model_name in models_name:
        model_config = MODEL_CONFIG[model_name]

        # load model
        # model = model_config["loader"](device=device)
        # model = model.eval()

        # # create model wrapper
        # model_wrapper = model_config["wrapper"](model)
        # # get sample input
        # sample_inputs = model_wrapper.get_sample_input(batch_size, height, width)

        # # run torch
        # with torch.inference_mode(), torch.autocast("cuda"):
        #     if isinstance(sample_inputs, tuple):
        #         torch_out = model_wrapper.get_model()(*sample_inputs)
        #     else:
        #         torch_out = model_wrapper.get_model()(sample_inputs)
        # model = model.to("cpu")
        # torch_out = to_numpy(torch_out)
        # torch.cuda.empty_cache()

        # load onnx graph
        onnx_path = os.path.join(base_path, model_name + ".opt", "model.onnx")
        print("checking graph structure")
        onnx.checker.check_model(onnx_path)
        print("graph structure valid")

        from google.protobuf.json_format import MessageToDict

        onnx_model = onnx.load(onnx_path)
        print("inputs")
        for _input in onnx_model.graph.input:
            print("  " + str(MessageToDict(_input)))

        print("outputs")
        for _input in onnx_model.graph.output:
            print("  " + str(MessageToDict(_input)))

        # load onnx graph
        ort_session = onnxruntime.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )

        print("inputs")
        for _input in ort_session.get_inputs():
            print("  " + str(_input))

        print("outputs")
        for _output in ort_session.get_outputs():
            print("  " + str(_output))

        # compute ONNX Runtime output prediction
        inputs = {}
        if isinstance(sample_inputs, tuple):
            for name, sample_input in zip(model_wrapper.get_input_names(), sample_inputs):
                # bind_input(sample_input, name, io_binding)
                inputs[name] = to_numpy(sample_input)

        elif isinstance(sample_inputs, torch.Tensor):
            # bind_input(sample_inputs, model_wrapper.get_input_names()[0], io_binding)
            inputs[model_wrapper.get_input_names()[0]] = to_numpy(sample_inputs)
        else:
            raise NotImplementedError()

        ort_output = ort_session.run(model_wrapper.get_output_names(), inputs)

        try:
            avg_err = np.mean(np.abs(torch_out - ort_output[0]))
            assert avg_err < 1e-3
            print(f"{model_name} is fine")
        except Exception:
            print(f"model {model_name} error: {avg_err}")
        print()
        # assert np.sum(np.abs(torch_out - ort_output[0]) > 1e-2) == 0

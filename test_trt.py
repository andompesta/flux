import os
import numpy as np
import torch
import tensorrt as trt

from cuda import cudart
from src.flux.trt.wrappers import BaseWrapper, AEWrapper, CLIPWrapper, FluxWrapper, T5Wrapper


from flux.util import load_ae, load_clip, load_flow_model, load_t5

TRT_LOGGER = trt.Logger()

MODEL_CONFIG = {
    "ae": {
        "wrapper": AEWrapper,
        "loader": lambda device: load_ae(device=device, name="flux-schnell"),
    },
    "clip": {
        "wrapper": CLIPWrapper,
        "loader": lambda device: load_clip(device=device),
    },
    "t5": {
        "wrapper": lambda x: T5Wrapper(x),
        "loader": lambda device: load_t5(device=device, max_length=256),
    },
    "transformer": {
        "wrapper": lambda x: FluxWrapper(x, fp16=True),
        "loader": lambda device: load_flow_model(device=device, name="flux-schnell"),
    },
}


# Map of TensorRT dtype -> torch dtype
trt_to_torch_dtype_dict = {
    trt.DataType.BOOL: torch.bool,
    trt.DataType.UINT8: torch.uint8,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.HALF: torch.float16,
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.BF16: torch.bfloat16,
}

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
    device = torch.device("cuda")
    batch_size = 2
    width: int = 1360
    height: int = 768

    engine_dir = "/workspace/data/flux/engine_test"
    models_name = [
        # "ae",
        # "clip",
        "t5",
        # "transformer",
    ]

    for model_name in models_name:
        model_config = MODEL_CONFIG[model_name]

        # load model
        model = model_config["loader"](device=device)
        model = model.eval()

        # create model wrapper
        model_wrapper: BaseWrapper = model_config["wrapper"](model)
        # get sample input
        sample_inputs = model_wrapper.get_sample_input(batch_size, height, width)

        # run torch
        with torch.inference_mode():
            if isinstance(sample_inputs, tuple):
                torch_out = model_wrapper.get_model()(*sample_inputs)
            else:
                torch_out = model_wrapper.get_model()(sample_inputs)
        model = model.to("cpu")
        torch_out = to_numpy(torch_out)
        torch.cuda.empty_cache()

        # load trt
        engine_path = os.path.join(
            engine_dir,
            model_name + ".trt" + trt.__version__ + ".plan",
        )
        tensors = {}

        feed_dict = {}
        shape_dict = model_wrapper.get_shape_dict(batch_size, height, width)
        if isinstance(sample_inputs, tuple):
            for name, sample_input in zip(model_wrapper.get_input_names(), sample_inputs):
                feed_dict[name] = sample_input
        elif isinstance(sample_inputs, torch.Tensor):
            feed_dict[model_wrapper.get_input_names()[0]] = sample_inputs
        else:
            raise NotImplementedError()

        with load_engine(engine_path=engine_path) as engine:
            stream = cudart.cudaStreamCreate()[1]

            max_device_memory = engine.device_memory_size_v2
            _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
            context = engine.create_execution_context_without_device_memory()
            context.device_memory = shared_device_memory

            for tensor_idx in range(engine.num_io_tensors):
                tensor_name = engine.get_tensor_name(tensor_idx)
                # tensor_shape = engine.get_tensor_shape(tensor_name)
                shape = shape_dict[tensor_name]

                if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    context.set_input_shape(tensor_name, shape)

                dtype = trt_to_torch_dtype_dict[engine.get_tensor_dtype(tensor_name)]
                tensor = torch.empty(
                    shape,
                    dtype=dtype,
                ).to(device=device)
                tensors[tensor_name] = tensor

            for name, buf in feed_dict.items():
                tensors[name].copy_(buf)

            for name, tensor in tensors.items():
                context.set_tensor_address(name, tensor.data_ptr())

            noerror = context.execute_async_v3(stream)
            if not noerror:
                raise ValueError(f"ERROR: inference failed.")

            trt_output = tensors[model_wrapper.get_output_names()[0]].clone().cpu().numpy()

        try:
            avg_err = np.mean(np.abs(torch_out - trt_output))
            assert avg_err < 1e-3
            print(f"{model_name} is fine")
        except Exception as e:
            print(f"model {model_name} error: {avg_err}")

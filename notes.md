1. why do I need to download the condition models twice:
    - they are part of the flux model repository https://huggingface.co/black-forest-labs/FLUX.1-schnell
    - they got downloaded by the load_model function https://github.com/black-forest-labs/flux/blob/87f6fff727a377ea1c378af692afb41ae84cbe04/src/flux/util.py#L129

2. how to handle autocast before variational autoencoder
3. how to handle pinned dependencies
4. is it possible to modify `cli` inference script
5. how to handle un-optimized graph as Flux model implementation is different from Diffuser

# Onnx export status

|      | **CLIP** |      | **AE** |      | **T5** not optim |      | **T5** |       |**Flux**|       |
|:----:|:--------:|:----:|:------:|:----:|:----------------:|:----:|--------|-------|--------|-------|
|      |    CPU   | CUDA |   CPU  | CUDA |        CPU       | CUDA |   CPU  |  CUDA |   CPU  |  CUDA |
| Fp16 |   1e-2   | 1e-2 |  1e-2  |  nan |       1e-2       | 1e-2 |  1e-2  | ?     |  1e-2  |    ?  |
| Fp32 |   1e-4   | 1e-2 |  1e-2  |  nan |        OOM       |  OOM |   OOM  |  OOM  |   OOM  |  OOM  |


1) Not clear why AE gives nan on CUDA ?
2) T5 optim does not work due to [cleanup steps](https://gitlab-master.nvidia.com/TensorRT/Public/oss/-/blame/release/10.5/demo/Diffusion/models.py#L530), removing it makes the model work.
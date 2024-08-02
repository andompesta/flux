# FLUX
by Black Forest Labs: https://blackforestlabs.ai

![grid](assets/grid.jpg)

This repo contains minimal inference code to run text-to-image and image-to-image with our Flux latent rectified flow transformers.

### Inference partners

We are happy to partner with [Replicate](https://replicate.com/) and [FAL](https://fal.ai/). You can sample our models using their services.
Below we list relevant links.

Replicate:
- https://replicate.com/collections/flux
- https://replicate.com/black-forest-labs/flux-pro
- https://replicate.com/black-forest-labs/flux-dev
- https://replicate.com/black-forest-labs/flux-schnell

FAL:
- https://fal.ai/models/fal-ai/flux-pro
- https://fal.ai/models/fal-ai/flux/dev
- https://fal.ai/models/fal-ai/flux/schnell


## Local installation

```bash
cd $HOME && git clone https://github.com/black-forest-labs/flux
cd $HOME/flux
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e '.[all]'
```

### Models

We are offering three models:
- `FLUX.1 [pro]` the base model, available via API
- `FLUX.1 [dev]` guidance-distilled variant
- `FLUX.1 [schnell]` guidance and step-distilled variant

| Name   | HuggingFace repo   | License    | md5sum    |
|-------------|-------------|-------------|-------------|
| `FLUX.1 [schnell]` | https://huggingface.co/black-forest-labs/FLUX.1-schnell | [apache-2.0](model_licenses/LICENSE-FLUX1-schnell) | a9e1e277b9b16add186f38e3f5a34044 |
| `FLUX.1 [dev]` | https://huggingface.co/black-forest-labs/FLUX.1-dev| [FLUX.1-dev Non-Commercial License](model_licenses/LICENSE-FLUX1-dev) | a6bd8c16dfc23db6aee2f63a2eba78c0  |
| `FLUX.1 [pro]` | Only available in our API. |

The weights of the autoencoder are also released under [apache-2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) and can be found in either of the two HuggingFace repos above. They are the same for both models.


## Usage

The weights will be downloaded automatically from HuggingFace once you start one of the demos. To download `FLUX.1 [dev]`, you will need to be logged in, see [here](https://huggingface.co/docs/huggingface_hub/guides/cli#huggingface-cli-login).
If you have downloaded the model weights manually, you can specify the downloaded paths via environment-variables:
```bash
export FLUX_SCHNELL=<path_to_flux_schnell_sft_file>
export FLUX_DEV=<path_to_flux_dev_sft_file>
export AE=<path_to_ae_sft_file>
```

For interactive sampling run
```bash
python -m flux --name <name> --loop
```
Or to generate a single sample run
```bash
python -m flux --name <name> \
  --height <height> --width <width> \
  --prompt "<prompt>"
```

We also provide a streamlit demo that does both text-to-image and image-to-image. The demo can be run via
```bash
streamlit run demo_st.py
```

We also offer a Gradio-based demo for an interactive experience. To run the Gradio demo:
```bash
python demo_gr.py --model flux-schnell --device cuda
```

Options:
- `--model`: Choose the model to use (options: "flux-schnell", "flux-dev")
- `--device`: Specify the device to use (default: "cuda" if available, otherwise "cpu")
- `--offload`: Offload model to CPU when not in use
- `--share`: Create a public link to your demo

To run the demo with the dev model and create a public link:

```bash
python -m demo_gr.py --model flux-dev --share
```

A live Gradio demo is also available on [Hugging Face](https://huggingface.co/black-forest-labs) based on 🧨 diffusers

## API usage

Our API offers access to the pro model. It is documented here:
[docs.bfl.ml](https://docs.bfl.ml/).

In this repository we also offer an easy python interface. To use this, you
first need to register with the API on [api.bfl.ml](https://api.bfl.ml/), and
create a new API key.

To use the API key either run `export BFL_API_KEY=<your_key_here>` or provide
it via the `api_key=<your_key_here>` parameter. Is is also expected that you
have installed the package as above.

Usage from python:
```python
from flux.api import ImageRequest

# this will create an api request directly but not block until the generation is finished
request = ImageRequest("A beautiful beach")
# or: request = ImageRequest("A beautiful beach", api_key="your_key_here")

# any of the following will block until the generation is finished
request.url
# -> https:<...>/sample.jpg
request.bytes
# -> b"..." bytes for the generated image
request.save("outputs/api.jpg")
# saves the sample to local storage
request.image
# -> a PIL image
```

Usage from the command line:
```bash
$ python -m flux.api --prompt="A beautiful beach" url
https:<...>/sample.jpg

# generate and save the result
$ python -m flux.api --prompt="A beautiful beach" save outputs/api

# open the image directly
$ python -m flux.api --prompt="A beautiful beach" image show
```

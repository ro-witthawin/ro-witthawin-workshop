# DiffusionGemma Workshop

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ro-witthawin/ro-witthawin-workshop/blob/main/workshops/DiffusionGemma/DiffusionGemma.ipynb)

Hands-on notebook for running `google/diffusiongemma-26B-A4B-it` with Hugging Face Transformers and observing how a diffusion language model iteratively refines generated text.

## What You Will Learn

The notebook demonstrates how to:

- load DiffusionGemma with `DiffusionGemmaForBlockDiffusion`
- prepare chat-style input with `AutoProcessor`
- run generation with `model.generate`
- stream intermediate diffusion drafts with `TextDiffusionStreamer`
- display each refinement step inside a notebook
- decode and render the final generated response as Markdown

## Files

- `DiffusionGemma.ipynb` - main workshop notebook
- `requirements.txt` - Python dependencies for running the notebook

## Main Stack

- Model: `google/diffusiongemma-26B-A4B-it`
- Runtime: Python 3.12 notebook environment
- Core library: `transformers`
- Notebook display: `IPython.display`
- Recommended environment: Google Colab or a local GPU workstation with sufficient memory

## Quick Start in Google Colab

1. Open the notebook with the Colab badge above.
2. Select a GPU runtime:

```text
Runtime -> Change runtime type -> Hardware accelerator -> GPU
```

3. Install or upgrade the notebook dependencies if your runtime does not already include a Transformers version with DiffusionGemma support:

```python
%pip install -U torch "transformers>=5.11.0" accelerate
```

4. Run the notebook cells from top to bottom.
5. Change the prompt in the `message` cell and rerun the processing and generation cells.

## Local Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r workshops/DiffusionGemma/requirements.txt
```

Start Jupyter:

```bash
jupyter notebook workshops/DiffusionGemma/DiffusionGemma.ipynb
```

The model is large, so local runs require a GPU setup with enough available memory. If loading fails because of memory limits, use a larger Colab runtime or a machine with more GPU memory.

## Workshop Flow

Run the notebook sections in order:

1. Import `DiffusionGemmaForBlockDiffusion` and `AutoProcessor`.
2. Set `MODEL_ID` to `google/diffusiongemma-26B-A4B-it`.
3. Load the processor and model with automatic dtype and device placement.
4. Define a chat prompt.
5. Apply the chat template and move tensors to the model device.
6. Create `LiveDiffusionStreamer` to display intermediate draft steps.
7. Generate up to `4096` new tokens with the streamer enabled.
8. Decode the output and render the final result as Markdown.

## Prompt Example

The notebook starts with:

```python
message = [
    {"role": "user", "content": "What is diffusion llm?"}
]
```

Try replacing the prompt with workshop questions such as:

```python
message = [
    {"role": "user", "content": "Explain diffusion language models in three concise bullet points."}
]
```

## Notes

- The first model load may take time because the checkpoint must be downloaded.
- `device_map="auto"` lets Transformers place model components on available hardware.
- `dtype="auto"` uses the model and hardware defaults selected by Transformers.
- `TextDiffusionStreamer` is used to reveal draft refinement steps before the final batch is displayed.

## Troubleshooting

### `ImportError` for DiffusionGemma classes

Upgrade Transformers:

```bash
pip install -U "transformers>=5.11.0"
```

Restart the notebook kernel after upgrading.

### Out-of-memory error

Use a runtime with more GPU memory, close other GPU processes, or reduce `max_new_tokens` while testing:

```python
output = model.generate(
    **input_ids,
    max_new_tokens=512,
    streamer=streamer,
)
```

### Slow first run

The first run downloads the model and initializes GPU memory. Later runs are usually faster once the checkpoint is cached.

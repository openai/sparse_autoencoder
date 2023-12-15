# Sparse autoencoder for GPT2 small

This repository hosts a sparse autoencoder trained on the GPT2-small model's activations.
The autoencoder's purpose is to expand the MLP layer activations into a larger number of dimensions,
providing an overcomplete basis of the MLP activation space. The learned dimensions have been
shown to be more interpretable than the original MLP dimensions.

### Install

```sh
pip install git+https://github.com/openai/sparse_autoencoder.git
```

### Example usage

```py
import torch
import blobfile as bf
import transformer_lens
import sparse_autoencoder

# Load the autoencoder
layer_index = 0  # in range(12)
autoencoder_input = ["mlp_post_act", "resid_delta_mlp"][0]
filename = f"az://openaipublic/sparse-autoencoder/gpt2-small/{autoencoder_input}/autoencoders/{layer_index}.pt"
with bf.BlobFile(filename, mode="rb") as f:
    state_dict = torch.load(f)
    autoencoder = sparse_autoencoder.Autoencoder.from_state_dict(state_dict)

# Extract neuron activations with transformer_lens
model = transformer_lens.HookedTransformer.from_pretrained("gpt2", center_writing_weights=False)
prompt = "This is an example of a prompt that"
tokens = model.to_tokens(prompt)  # (1, n_tokens)
print(model.to_str_tokens(tokens))
with torch.no_grad():
    logits, activation_cache = model.run_with_cache(tokens, remove_batch_dim=True)
if autoencoder_input == "mlp_post_act":
    input_tensor = activation_cache[f"blocks.{layer_index}.mlp.hook_post"]  # (n_tokens, n_neurons)
elif autoencoder_input == "resid_delta_mlp":
    input_tensor = activation_cache[f"blocks.{layer_index}.hook_mlp_out"]  # (n_tokens, n_residual_channels)

# Encode neuron activations with the autoencoder
device = next(model.parameters()).device
autoencoder.to(device)
with torch.no_grad():
    latent_activations = autoencoder.encode(input_tensor)  # (n_tokens, n_latents)
```

### Autoencoder settings

- Model used: "gpt2-small", 12 layers
- Autoencoder architecture: see `model.py`
- Autoencoder input: "mlp_post_act" (3072 dimensions) or "resid_delta_mlp" (768 dimensions)
- Number of autoencoder latents: 32768
- Loss function: see `loss.py`
- Number of training tokens: ~64M
- L1 regularization strength: 0.01

### Data files

- `autoencoder_input` is in ["mlp_post_act", "resid_delta_mlp"]
- `layer_index` is in range(12) (GPT2-small)
- `latent_index` is in range(32768)

Autoencoder files:
`az://openaipublic/sparse-autoencoder/gpt2-small/{autoencoder_input}/autoencoders/{layer_index}.pt`

NeuronRecord files:
`az://openaipublic/sparse-autoencoder/gpt2-small/{autoencoder_input}/collated_activations/{layer_index}/{latent_index}.json`

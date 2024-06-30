BASE_URL = "az://openaipublic/sparse-autoencoder/gpt2-small"
VALID_LOCATIONS_V1_V4 = ["mlp_post_act", "resid_delta_mlp"]
VALID_LOCATIONS_V5 = ["resid_delta_attn", "resid_delta_mlp", "resid_post_attn", "resid_post_mlp"]

def generate_url(location: str, layer_index: int, version: str) -> str:
    return f"{BASE_URL}/{location}_{version}/autoencoders/{layer_index}.pt"

def v1(location, layer_index):
    """
    Details:
    - Number of autoencoder latents: 32768
    - Number of training tokens: ~64M
    - Activation function: ReLU
    - L1 regularization strength: 0.01
    - Layer normed inputs: false
    - NeuronRecord files:
        `az://openaipublic/sparse-autoencoder/gpt2-small/{location}/collated_activations/{layer_index}/{latent_index}.json`
    """
    assert location in VALID_LOCATIONS_V1_V4, f"Invalid location: {location}"
    assert layer_index in range(12), f"Invalid layer_index: {layer_index}"
    return f"{BASE_URL}/{location}/autoencoders/{layer_index}.pt"

def v4(location, layer_index):
    """
    Details:
    same as v1
    """
    assert location in VALID_LOCATIONS_V1_V4, f"Invalid location: {location}"
    assert layer_index in range(12), f"Invalid layer_index: {layer_index}"
    return generate_url(location, layer_index, "v4")

def v5_32k(location, layer_index):
    """
    Details:
    - Number of autoencoder latents: 2**15 = 32768
    - Number of training tokens:  TODO
    - Activation function: TopK(32)
    - L1 regularization strength: n/a
    - Layer normed inputs: true
    """
    assert location in VALID_LOCATIONS_V5, f"Invalid location: {location}"
    assert layer_index in range(12), f"Invalid layer_index: {layer_index}"
    # note: it's actually 2**15 and 2**17 ~= 131k
    return generate_url(location, layer_index, "v5_32k")

def v5_128k(location, layer_index):
    """
    Details:
    - Number of autoencoder latents: 2**17 = 131072
    - Number of training tokens: TODO
    - Activation function: TopK(32)
    - L1 regularization strength: n/a
    - Layer normed inputs: true
    """
    assert location in VALID_LOCATIONS_V5, f"Invalid location: {location}"
    assert layer_index in range(12), f"Invalid layer_index: {layer_index}"
    # note: it's actually 2**15 and 2**17 ~= 131k
    return generate_url(location, layer_index, "v5_128k")

# NOTE: we have larger autoencoders (up to 8M, with varying n and k) trained on layer 8 resid_post_mlp
# we may release them in the future

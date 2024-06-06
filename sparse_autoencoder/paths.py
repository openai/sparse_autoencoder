
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
    assert location in ["mlp_post_act", "resid_delta_mlp"]
    assert layer_index in range(12)
    return f"az://openaipublic/sparse-autoencoder/gpt2-small/{location}/autoencoders/{layer_index}.pt"

def v4(location, layer_index):
    """
    Details:
    same as v1
    """
    assert location in ["mlp_post_act", "resid_delta_mlp"]
    assert layer_index in range(12)
    return f"az://openaipublic/sparse-autoencoder/gpt2-small/{location}_v4/autoencoders/{layer_index}.pt"

def v5_32k(location, layer_index):
    """
    Details:
    - Number of autoencoder latents: 2**15 = 32768
    - Number of training tokens:  TODO
    - Activation function: TopK(32)
    - L1 regularization strength: n/a
    - Layer normed inputs: true
    """
    assert location in ["resid_delta_attn", "resid_delta_mlp", "resid_post_attn", "resid_post_mlp"]
    assert layer_index in range(12)
    # note: it's actually 2**15 and 2**17 ~= 131k
    return f"az://openaipublic/sparse-autoencoder/gpt2-small/{location}_v5_32k/autoencoders/{layer_index}.pt"

def v5_128k(location, layer_index):
    """
    Details:
    - Number of autoencoder latents: 2**17 = 131072
    - Number of training tokens: TODO
    - Activation function: TopK(32)
    - L1 regularization strength: n/a
    - Layer normed inputs: true
    """
    assert location in ["resid_delta_attn", "resid_delta_mlp", "resid_post_attn", "resid_post_mlp"]
    assert layer_index in range(12)
    # note: it's actually 2**15 and 2**17 ~= 131k
    return f"az://openaipublic/sparse-autoencoder/gpt2-small/{location}_v5_128k/autoencoders/{layer_index}.pt"

# NOTE: we have larger autoencoders (up to 8M, with varying n and k) trained on layer 8 resid_post_mlp
# we may release them in the future

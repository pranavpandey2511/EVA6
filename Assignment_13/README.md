<br/>
<h1 align="center">Session 10: ViT - An Image is worth 16x16 words
<br/>
<!-- toc -->
    <br>
    
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/RajamannarAanjaram/badges/)
[![Awesome Badges](https://img.shields.io/badge/badges-awesome-green.svg)](https://github.com/RajamannarAanjaram/badges)
    <br>
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/RajamannarAanjaram/)

### Contributors

<p align="center"> <b>Team - 6</b> <p>
    
| <centre>Name</centre> | <centre>Mail id</centre> | 
| ------------ | ------------- |
| <centre>Amit Agarwal</centre>         | <centre>amit.pinaki@gmail.com</centre>    |
| <centre>Pranav Panday</centre>         | <centre>pranavpandey2511@gmail.com</centre>    |
| <centre>Rajamannar A K</centre>         | <centre>rajamannaraanjaram@gmail.com</centre>    |
| <centre>Sree Latha Chopparapu</centre>         | <centre>sreelathaemail@gmail.com</centre>    |\\

<!-- toc -->

### Intution of Classes employed in VIT

### Class :: PatchEmbeddings(nn.Module)
Conversion of an image size to the patch embeddings
1. Takes an input image and check if its a tuple and converts, if not one. 
2. Calculates the number of patches required.
3. Captures the batch size, no.of channels and image dimensions from the pixel_vlues.shape 
4. Check for the proper dimensions of the image.
5. Projects the image to a required size(passing it to a 2D network), followed by flatten and transpose.

### class :: ViTEmbeddings(nn.Module)
Construct the CLS token, position and patch embeddings.
1. ViTEmbeddings expects a predefined configuration file with different attributes of image and batch
2. Construction of CLS tokens, with random numbers of size mentioned in the config.hidden_size
3. Construction of Patchembeddings with image_size, patch_size, num_channels, embed_dim as configured in the config.
4. Construction of position embeddings for num_patches+1
5. Concatanation of cls_tokens, patch_embeddings.
6. Add the postion_embeddings to all the patch_embeddings.
7. returns the embeddings.

### class :: ViTConfig
Construction of Confiuration Object.

Exemplification of Config is as follows:


attention_probs_dropout_prob: 0.0,
 
 hidden_act: 'gelu',

 hidden_dropout_prob: 0.0,

 hidden_size: 768,

 image_size: 224,

 initializer_range: 0.02,

 intermediate_size: 3072,

 layer_norm_eps: 1e-12,

 num_attention_heads: 12,

 num_channels: 3,

 num_hidden_layers: 12,

 patch_size: 16

 ### class :: ViTSelfAttention(nn.Module)
Costruction of self Attention matrix.
1. Check if the hidden_size is divisible by the num_attention_heads.
2. Initialising the num_attention_heads, attention_head_size, all_head_size as configured in the config.
3. Creating the query, key and Value matrices of size hidden_size, all_head_size
4. Reshaping the matrices to required sizes by Permute operation.
5. Make the query, key and Value layers.
6. Multiplication of query and key matrices to get the attention scores.
7. Normalize the attention scores.
8. Making the context_layer which is a Product of attention_probs and value_layer.
9. Reshaping the context_layer by permute followed by "Contiguous"

### class :: ViTSelfOutput(nn.Module)
This class specifies the Linear layer block.
1. Making the dense layers, takes hidden_size and gives layer of hidden_size.

### class :: ViTAttention(nn.Module)
Class Specifying the ViT Attention layer.
1. get the ViTSelfAttention object, ViTSelfOutput and pruned_heads(Required to remove the unwanted heads).
2. get the self_outputs from the FC layer.
3. Get the attention_output layer
4. Add the attentions if we return the outputs

### class ViTIntermediate(nn.Module)
Specifies the ViTintermediate Linear layer
1. Intakes the hidden_states and dense follwed by gelu activation

### class :: ViTOutput(nn.Module):
This class specifies the ViTOutput layer
1. Expansion of the intermediate layers

### class :: ViTLayer(nn.Module)
This corresponds to the Block class in the timm implementation
1. Get the ViTAttention, ViTIntermediate and ViTOutput objects, layernorm_before and layer_norm after.
2. Get the self_attention_outputs which intakes the layernorm_before, head_mask and output_attentions.
3. Add self attentions 
4. Add the skip connections.
5. Layernorm is also applied after self-attention
6. return the outputs.

### class ViTEncoder(nn.Module):
This Class specifies the overall VIT layer
1. Get all the hidden states
2. Layer outputs are obtained through the layer module using hidden_states, layer_head_masak, ouput_attention
3. Return all the hidden states required.

### class ViTPooler(nn.Module):
This class specifies pooling the model.
1. We "pool" the model by simply taking the hidden state corresponding to the first token 

### class ViTModel()
This class specifies the actual model for ViT training

# ComfyUI wrapper nodes for Pyramid-Flow


https://github.com/user-attachments/assets/9592bfed-9cf8-438a-b1e6-969d6b13ec12


todo
- refactor to use comfy text encoding instead
- optimize memory use

Besides text encoder, which can peak at ~12GB VRAM use, this should run at 9-10GB VRAM when using 1280x768.
With fp8 and 384p model it can fit under 6GB too. Note that these tests were done on 4090, older cards may not support every optimization.

Resolutions outside the model defaults perform poorly.

Model loading has not been optimized at all yet, currently needs everything (choose either of the transformers) from here:

https://huggingface.co/rain1011/pyramid-flow-sd3/tree/main

to:

`ComfyUI/models/pyramidflow/pyramid-flow-sd3`

So that the directory structure is as follows:
```
\ComfyUI\models\pyramidflow\pyramid-flow-sd3
├───causal_video_vae
│       config.json
│       diffusion_pytorch_model.safetensors
│
├───diffusion_transformer_384p
│       config.json
│       diffusion_pytorch_model.safetensors
│
├───diffusion_transformer_768p
│       config.json
│       diffusion_pytorch_model.safetensors
│
├───text_encoder
│       config.json
│       model.safetensors
│
├───text_encoder_2
│       config.json
│       model.safetensors
│
├───text_encoder_3
│       config.json
│       model-00001-of-00002.safetensors
│       model-00002-of-00002.safetensors
│       model.safetensors.index.json
│
├───tokenizer
│       merges.txt
│       special_tokens_map.json
│       tokenizer_config.json
│       vocab.json
│
├───tokenizer_2
│       merges.txt
│       special_tokens_map.json
│       tokenizer_config.json
│       vocab.json
│
└───tokenizer_3
        special_tokens_map.json
        spiece.model
        tokenizer.json
        tokenizer_config.json
```

Original repo: https://github.com/jy0205/Pyramid-Flow

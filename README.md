# ComfyUI wrapper nodes for Pyramid-Flow


https://github.com/user-attachments/assets/ab66fed8-4a91-4b45-9c08-77086257833f


todo
- refactor to use comfy text encoding instead
- optimize memory use

Besides text encoder, this should already run at 10-12GB VRAM when using 1280x768.
With fp8 and 384p model it can fit under 10GB too.

Model loading has not been optimized at all yet, currently needs everything (choose either of the transformers) from here:

https://huggingface.co/rain1011/pyramid-flow-sd3/tree/main

to:

`ComfyUI/models/pyramidflow/pyramid-flow-sd3`

So that the directory structure is as follows:
```
\ComfyUI\models\pyramidflow\pyramid-flow-sd3
├───causal_video_vae
│       config.json
│       diffusion_pytorch_model.bin
│
├───diffusion_transformer_384p
│       config.json
│       diffusion_pytorch_model.bin
│
├───diffusion_transformer_768p
│       config.json
│       diffusion_pytorch_model.bin
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

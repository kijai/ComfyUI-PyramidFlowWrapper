# ComfyUI wrapper nodes for Pyramid-Flow

# NOT FULLY FUNCTIONAL YET

![image](https://github.com/user-attachments/assets/f863b596-6417-4d02-91ba-f9143f696d68)

todo
- refactor to use comfy text encoding instead
- optimize memory use

Besides text encoder, this should already run at 10-12GB VRAM

Model loading has not been optimized at all yet, currently needs everything from here:

https://huggingface.co/rain1011/pyramid-flow-sd3/tree/main

to:

`ComfyUI/models/pyramidflow/pyramid-flow-sd3`

Original repo: https://github.com/jy0205/Pyramid-Flow

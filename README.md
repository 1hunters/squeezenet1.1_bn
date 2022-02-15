# torch_squeezenet_bn_imagenet
Squeezenet with BN layers pytorch implementation for imagenet

Details:

Batch size (per GPU) = 512, Learning rate = 0.2, Epochs = 160, Weight decay = 4e-5.

More details can be found in 'config_squeeze.yaml'.

Training command:
```
python -m torch.distributed.launch --nproc_per_node=NUM_GPUs entry.py config_squeeze.yaml
```


Benefits of inserting BN Layer:

More stable for Mixed-precision quantization.


# Latest inference checkpoint

`best.pt` contains only the Transformer architecture, configuration and learned state. Training optimizer/scheduler state, metrics and private development paths have been removed. Parameter tensors are identical to the retained latest development checkpoint.

Load with `torch.load(path, map_location="cpu", weights_only=True)` or `alpha2048.value_models.load_value_model`. Hash, size and evaluation context are in manifest.json. The 19-game development evaluation is small and is not a performance guarantee. Training data and the training pipeline are not distributed.

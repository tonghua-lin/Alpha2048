# Model and search

## Network

The model has 206,849 parameters: four Transformer encoder layers, width 64, four attention heads and a 256-unit feed-forward block. Each board cell is encoded by tile exponent (empty=0, 2=1, 4=2), plus row and column embeddings. Mean pooling and an MLP produce a scalar afterstate value. The stored target mean and standard deviation restore the training score scale.

The public checkpoint contains the architecture/configuration and learned state only. Optimizer, scheduler and training logs are excluded. It is a value estimator, not a direct predictor of the final game score.

## Search

Let T(s,a) denote the board after a slide/merge and before a random tile spawn, and V denote the model.

- Depth 0: Q(s,a) = V(T(s,a)).
- Depth 1: enumerate every possible spawn (2 with probability 0.9, 4 with 0.1; uniform empty location), maximize over the following legal move, then average with spawn probabilities.
- Depth 2: repeat another spawn/decision layer before evaluating the frontier.

Illegal actions are masked. Terminal positions have value -10. Immediate merge rewards are not added to the learned values. Ties within 1e-6 use Up, Right, Down, Left priority. Batch limits bound individual expansion tensors, not all memory across the tree.

The UI converts action values to relative recommendations using softmax temperature 0.25. Rounded legal percentages sum to 100.0%. Game auto-play chooses the highest raw value; it does not sample from those percentages.

## Training context and evaluation limits

The initial model fitted a mixed-depth structural heuristic. Later iterations fitted the previous model's depth-1 search targets and continued from its parameters. Training split whole games into train/validation and used eight board symmetries; runtime inference does not ensemble those symmetries.

The latest checkpoint's recorded partial evaluation contained 19 games, mean score 142,136.84 and an 8192 reach rate of 84.21%. This is a small development evaluation, not a paired benchmark or a general performance guarantee. The original training pipeline and data are not distributed here, so these results are not fully reproducible from this repository alone.

## Runtime

Both interfaces use the same persistent-worker inference service and latest checkpoint. CPU uses four PyTorch threads; OCR uses two ONNX Runtime threads. Search can be cancelled at batch boundaries. Source builds support compatible CUDA hardware, while the portable CPU build deliberately excludes CUDA libraries.

Screenshots are processed locally. The recognition pipeline crops a selected 4×4 grid, checks empty cells, converts text to grayscale, normalizes polarity and performs recognition-only OCR. Stable frames are required before publishing a recommendation. Confidence checks cannot eliminate every OCR error; inspect the displayed board when results look wrong.

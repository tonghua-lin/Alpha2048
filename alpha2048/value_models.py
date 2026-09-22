"""Architecture-aware checkpoint loading, including original Transformer files."""
import torch
from .value_transformer import ValueTransformer


def build_value_model(model_type='transformer', **config):
    classes = {'transformer': ValueTransformer}
    if model_type not in classes:
        raise ValueError(f'Unsupported value model: {model_type}')
    model = classes[model_type](**config)
    model.model_type = model_type
    return model


def load_value_model(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    # The first-generation files have no architecture field.
    model = build_value_model(checkpoint.get('model_type', 'transformer'),
                              **checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model'])
    return model.eval()

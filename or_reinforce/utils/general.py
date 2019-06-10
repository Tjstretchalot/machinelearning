"""Generally helpful functions"""
import torch
import os

def save_model(model, modelfile: str):
    """Saves the given model to the given file"""
    torch.save(model, modelfile)

def load_model(modelfile: str):
    """Loads the model from the given file"""
    return torch.load(modelfile)

def init_or_load_model(init_model, modelfile: str):
    """Inits the specified model using the specified function if there is
    no save available, otherwise loads the latest version"""
    basedir = os.path.dirname(modelfile)
    os.makedirs(basedir, exist_ok=True)

    if os.path.exists(modelfile):
        return load_model(modelfile)
    model = init_model()
    save_model(model, modelfile)
    return model

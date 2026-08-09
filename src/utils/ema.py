import torch
import copy

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        # Create a deep copy of the model
        self.ema_model = copy.deepcopy(model)
        # Freeze the EMA model (we only update it via moving average)
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            # Ensure EMA model is on the same device as the source model
            model_device = next(model.parameters()).device
            ema_device = next(self.ema_model.parameters()).device
            if model_device != ema_device:
                self.ema_model = self.ema_model.to(model_device)

            for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)
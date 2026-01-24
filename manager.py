import torch

class MOEManager:
    """
    basic wrapper class for tracking, storing, and aggregating auxiliary
    losses across multiple MoE layers in the model
    """

    def __init__(self, ortho_loss_start_frac=0):
        self.ortho_loss_start_frac = ortho_loss_start_frac
        self._values = {
            "aux_loss": [],
            "router_z_loss": [],
            "router_ortho_loss": [],
            "experts_ortho_loss": [],
            "gate_output_loss": [],
            "projs_diversity_loss": [],
            "drop_rate_per_ks": [],
        }
        self._start_frac_names = {
            "router_ortho_loss",
            "experts_ortho_loss",
            "gate_output_loss",
            "projs_diversity_loss",
        }

    def reset(self, name):
        self._values[name] = []

    def add(self, name, value):
        if name not in self._values:
            self._values[name] = []
        self._values[name].append(value)

    def aggregate(self, name):
        values = self._values.get(name, [])
        if name in self._start_frac_names:
            # If ortho_loss_start_frac = 0.25 and there are 8 moe layers, then 0.25*8 = 2.0, 
            # so start from layer 2, i.e. skip first two layers.
            # But usually we set ortho_loss_start_frac = 0, i.e. sum losses on all layers.
            start_layer = int(len(values) * self.ortho_loss_start_frac)
            values = values[start_layer:]
        if name == "drop_rate_per_ks":
            # drop_rate_per_ks is a list of tensors.
            values = torch.stack(values)
            return values.mean(dim=0) if values.numel() > 0 else None
        else:
            return sum(values)
    
MANAGER = MOEManager(ortho_loss_start_frac=0.)

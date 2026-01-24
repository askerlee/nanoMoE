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
        self._drop_rate_capacity = 32
        self._drop_rate_buffer = None
        self._drop_rate_size = 0
        self._start_frac_names = {
            "router_ortho_loss",
            "experts_ortho_loss",
            "gate_output_loss",
            "projs_diversity_loss",
        }

    def reset(self, name):
        if name == "drop_rate_per_ks":
            self._drop_rate_size = 0
            return
        self._values[name] = []

    def add(self, name, value):
        if name == "drop_rate_per_ks":
            if self._drop_rate_buffer is None:
                self._drop_rate_buffer = torch.empty(
                    (self._drop_rate_capacity, value.shape[0]),
                    device=value.device,
                    dtype=value.dtype,
                )
            new_size = self._drop_rate_size + 1
            self._drop_rate_buffer[self._drop_rate_size:new_size].copy_(value)
            self._drop_rate_size = new_size
            return
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
            if self._drop_rate_buffer is None or self._drop_rate_size == 0:
                return None
            values = self._drop_rate_buffer[:self._drop_rate_size]
            return values.mean(dim=0)
        else:
            return sum(values)
    
MANAGER = MOEManager(ortho_loss_start_frac=0.)

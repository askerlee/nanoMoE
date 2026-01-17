class MOEManager:
    """
    basic wrapper class for tracking, storing, and aggregating auxiliary
    losses across multiple MoE layers in the model
    """

    def __init__(self, ortho_loss_start_frac=0):
        self.aux_losses = []
        self.router_z_losses = []
        self.router_ortho_losses = []
        self.experts_ortho_losses = []
        self.gate_output_losses = []
        self.gate_diversity_losses = []
        self.ortho_loss_start_frac = ortho_loss_start_frac

    def reset_aux_loss(self):
        self.aux_losses = []
    
    def reset_router_z_loss(self):
        self.router_z_losses = []
    
    def reset_router_ortho_loss(self):
        self.router_ortho_losses = []
    
    def reset_experts_ortho_loss(self):
        self.experts_ortho_losses = []

    def reset_gate_output_loss(self):
        self.gate_output_losses = []

    def reset_gate_diversity_loss(self):
        self.gate_diversity_losses = []

    def add_aux_loss(self, loss):
        self.aux_losses.append(loss)
    
    def add_router_z_loss(self, loss):
        self.router_z_losses.append(loss)

    def add_router_ortho_loss(self, loss):
        self.router_ortho_losses.append(loss)

    def add_experts_ortho_loss(self, loss):
        self.experts_ortho_losses.append(loss)

    def add_gate_output_loss(self, loss):
        self.gate_output_losses.append(loss)

    def add_gate_diversity_loss(self, loss):
        self.gate_diversity_losses.append(loss)

    def aggregate_aux_loss(self):
        return sum(self.aux_losses)

    def aggregate_router_z_loss(self):
        return sum(self.router_z_losses)

    def aggregate_router_ortho_loss(self):
        # 0.25*8 = 2.0, so start from layer 2, i.e. skip first two layers
        start_layer = int(len(self.router_ortho_losses) * self.ortho_loss_start_frac)
        return sum(self.router_ortho_losses[start_layer:])
    
    def aggregate_experts_ortho_loss(self):
        # 0.25*8 = 2.0, so start from layer 2, i.e. skip first two layers
        start_layer = int(len(self.experts_ortho_losses) * self.ortho_loss_start_frac)
        return sum(self.experts_ortho_losses[start_layer:])
    
    def aggregate_gate_output_loss(self):
        start_layer = int(len(self.gate_output_losses) * self.ortho_loss_start_frac)
        return sum(self.gate_output_losses[start_layer:])

    def aggregate_gate_diversity_loss(self):
        start_layer = int(len(self.gate_diversity_losses) * self.ortho_loss_start_frac)
        return sum(self.gate_diversity_losses[start_layer:])
    
MANAGER = MOEManager(ortho_loss_start_frac=0.)

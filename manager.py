class MOEManager:
    """
    basic wrapper class for tracking, storing, and aggregating auxiliary
    losses across multiple MoE layers in the model
    """

    def __init__(self, ortho_loss_start_layer=0):
        self.aux_loss = []
        self.router_z_loss = []
        self.router_ortho_loss = []
        self.experts_ortho_loss = []
        self.ortho_loss_start_layer = ortho_loss_start_layer

    def reset_aux_loss(self):
        self.aux_loss = []
    
    def reset_router_z_loss(self):
        self.router_z_loss = []
    
    def reset_router_ortho_loss(self):
        self.router_ortho_loss = []
    
    def reset_experts_ortho_loss(self):
        self.experts_ortho_loss = []

    def add_aux_loss(self, loss):
        self.aux_loss.append(loss)
    
    def add_router_z_loss(self, loss):
        self.router_z_loss.append(loss)

    def add_router_ortho_loss(self, loss):
        self.router_ortho_loss.append(loss)

    def add_experts_ortho_loss(self, loss):
        self.experts_ortho_loss.append(loss)

    def aggregate_aux_loss(self):
        return sum(self.aux_loss)

    def aggregate_router_z_loss(self):
        return sum(self.router_z_loss)

    def aggregate_router_ortho_loss(self):
        return sum(self.router_ortho_loss[self.ortho_loss_start_layer:])
    
    def aggregate_experts_ortho_loss(self):
        return sum(self.experts_ortho_loss[self.ortho_loss_start_layer:])
    
MANAGER = MOEManager(ortho_loss_start_layer=4)

class Architect:
    def __init__(self, model, lr, wd):
        self.model = model
        self.optimizer = torch.optim.Adam(model.arch_parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=wd)

    def step(self):
        self.optimizer.step()
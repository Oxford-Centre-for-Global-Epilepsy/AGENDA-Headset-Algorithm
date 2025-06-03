class Controller:
    def __init__(self, model, architect, train_loader, valid_loader):
        self.model = model
        self.architect = architect
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def search(self, epochs):
        for epoch in range(epochs):
            # Normally you'd use both loaders to alternate updates
            for step, (x, y) in enumerate(self.train_loader):
                logits = self.model(x)
                loss = nn.functional.cross_entropy(logits, y)
                loss.backward()
                self.architect.step()
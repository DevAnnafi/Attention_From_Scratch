import torch.optim as optim

class WarmupScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self._compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _compute_lr(self):
        step = self.step_num
        return (self.d_model ** -0.5) * min(
            step ** -0.5,
            step * self.warmup_steps ** -1.5
        )

    def zero_grad(self):
        self.optimizer.zero_grad()


def get_optimizer_and_scheduler(model, d_model, warmup_steps=4000):
    optimizer = optim.Adam(
        model.parameters(),
        lr=0,           
        betas=(0.9, 0.98),
        eps=1e-9        
    )
    scheduler = WarmupScheduler(optimizer, d_model, warmup_steps)
    return optimizer, scheduler
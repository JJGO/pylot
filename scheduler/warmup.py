from contextlib import contextmanager
import warnings


@contextmanager
def momentum_correction(optim):
    if "momentum" not in optim.defaults:
        yield
        return
    lrs_old = [pg["lr"] for pg in optim.param_groups]
    momentum = optim.defaults["momentum"]
    yield
    for pg, lr_old in zip(optim.param_groups, lrs_old):
        lr = pg["lr"]
        if lr > lr_old:
            pg["momentum"] = lr / lr_old * momentum
        else:
            pg["momentum"] = momentum


class WarmupScheduler:
    def __init__(self, warmup_period, scheduler=None, last_step=0, skip=None):
        # Period is in Iterations
        # Skip is in epochs and says whether to skip the scheduler `skip` steps
        # so we linearly go to the correct target learning rate
        self.warmup_period = warmup_period
        self.scheduler = scheduler
        self.last_step = last_step
        if skip is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self.scheduler.step(skip - 1)
        self.target_lrs = [pg["lr"] for pg in self.scheduler.optimizer.param_groups]

    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        state_dict = {
            key: value for key, value in self.__dict__.items() if key != "scheduler"
        }
        if self.scheduler is not None:
            state_dict["scheduler"] = self.scheduler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.
        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict.pop("scheduler"))
        self.__dict__.update(state_dict)

    def step(self, epoch=None):
        if self.last_step < self.warmup_period:
            return
        if self.scheduler is not None:
            with momentum_correction(self.scheduler.optimizer):
                self.scheduler.step(epoch)

    def warmup_step(self, step=None):
        if self.last_step >= self.warmup_period:
            return

        if step is None:
            step = self.last_step + 1
        self.last_step = step

        with momentum_correction(self.scheduler.optimizer):
            for i, pg in enumerate(self.scheduler.optimizer.param_groups):
                omega = self.warmup_factor(step)
                pg["lr"] = self.target_lrs[i] * omega

    def warmup_factor(self, step):
        """ Computes Linear warmup
        """
        return min(1.0, step / self.warmup_period)

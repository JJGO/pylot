from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
from pylot.experiment import TrainExperiment
from pylot.util.pipes import quiet_std
from pylot.util import ThunderDict, MetricsDict
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


class DistTE(TrainExperiment):
    @property
    def is_master(self):
        return self.rank == 0

    def to_device(self):
        super().to_device()
        self.orig_model = self.model
        self.model = DDP(self.model)

    def build_dataloader(self):
        train_sampler = DistributedSampler(
            dataset=self.train_dataset,
            num_replicas=self.config["dist.world_size"],
            rank=self.rank,
            shuffle=True,
            drop_last=True,
        )

        val_sampler = DistributedSampler(
            dataset=self.val_dataset,
            num_replicas=self.config["dist.world_size"],
            rank=self.rank,
            shuffle=False,
            drop_last=False,
        )

        dl_cfg = self.config["dataloader"]
        self.train_dl = DataLoader(self.train_dataset, sampler=train_sampler, **dl_cfg)
        self.val_dl = DataLoader(self.val_dataset, sampler=val_sampler, **dl_cfg)

    def setup_distributed(self, rank):
        world_size = self.config["dist.world_size"]
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = self.port()
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    def checkpoint(self, *args, **kwargs):
        if self.is_master:
            super().checkpoint(*args, **kwargs)

    def run_callbacks(self, *args, **kwargs):
        if self.is_master:
            super().run_callbacks(*args, **kwargs)

    def port(self) -> str:
        nonce = self.metadata["nonce"]
        x = int.from_bytes(nonce.encode(), "big")
        port = 44_000 + (x % 10_000)
        return str(port)

    def run_distributed(self, rank):
        self.rank = rank
        self.setup_distributed(rank)
        if self.is_master:
            super().run()
        else:
            import tempfile

            # limit file writing
            with tempfile.TemporaryDirectory() as tmpdirname:
                self.metricsd = MetricsDict(tmpdirname)
                self.store = ThunderDict(tmpdirname + "/store")

                super().run()

    # def run_phase(self, phase, epoch):
    #     if ("val" in phase or "test" in phase) and not self.is_master:
    #         return
    #     super().run_phase(phase, epoch)

    def run(self):
        raise ValueError("Use exp.run_distributed instead")

        # TODO: timm warmup

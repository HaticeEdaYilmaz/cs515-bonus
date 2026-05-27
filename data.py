import torch
from torch.utils.data import Dataset, DataLoader
from config import Config


class MessageDataset(Dataset):
    """
    Each sample is a message m ∈ {0,...,7}^4  (0-indexed internally;
    the paper uses {1,...,8} but shifting by 1 has no effect on training).
    """
    def __init__(self, size: int, config: Config):
        torch.manual_seed(config.seed if size == config.train_size else config.seed + 1)
        self.data = torch.randint(0, config.vocab_size, (size, config.seq_len))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]   # (4,)  int64


def get_dataloaders(config: Config):
    train_ds = MessageDataset(config.train_size, config)
    val_ds   = MessageDataset(config.val_size,   config)

    train_dl = DataLoader(
        train_ds, batch_size=config.batch_size,
        shuffle=True,  num_workers=2, pin_memory=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=config.batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )
    return train_dl, val_dl

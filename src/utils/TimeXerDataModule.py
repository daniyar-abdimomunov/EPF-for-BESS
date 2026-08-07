from argparse import Namespace
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from timexer.data_provider.data_loader import Dataset_Custom

class TimeXerDataModule(LightningDataModule):
    def __init__(
            self,
            batch_size: int,
            data_path: str,
            embed: str,
            features: str, # one of ['M', 'MS', 'S']
            freq: str,
            num_workers: int,
            root_path: str,
            target: str, # name of target feature
            seq_len: int = 24 * 7,
            label_len: int = 24 * 2,
            pred_len: int = 24 * 1,
            **kwargs
    ):
        """
        LightningDataModule wrapper for TimeXer Custom Dataset.
        """
        super().__init__()
        # load in data
        self.args = Namespace(**kwargs)
        self.root_path = root_path
        self.data_path = data_path
        self.target = target
        self.features = features
        self.data = None
        self.timeenc = embed == 'timeF'
        self.freq = freq
        self.data_stamp = None

        # Initialize the scaler inside the DataModule
        self.drop_last = False
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        self.num_workers = num_workers

        self.train = None
        self.val = None
        self.test = None

        self.scaler = None

        self.setup()

        return

    def setup(self, stage=None):
        self.train = Dataset_Custom(
            args=self.args,
            root_path=self.root_path,
            data_path=self.data_path,
            flag='train',
            size=[self.seq_len, self.label_len, self.pred_len],
            features=self.features,
            target=self.target,
            timeenc=self.timeenc,
            freq=self.freq,
        )
        self.val = Dataset_Custom(
            args=self.args,
            root_path=self.root_path,
            data_path=self.data_path,
            flag='val',
            size=[self.seq_len, self.label_len, self.pred_len],
            features=self.features,
            target=self.target,
            timeenc=self.timeenc,
            freq=self.freq,
        )
        self.test = Dataset_Custom(
            args=self.args,
            root_path=self.root_path,
            data_path=self.data_path,
            flag='test',
            size=[self.seq_len, self.label_len, self.pred_len],
            features=self.features,
            target=self.target,
            timeenc=self.timeenc,
            freq=self.freq,
        )
        self.scaler = self.train.scaler

        return

    def train_dataloader(self):
        shuffle_flag = True
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=shuffle_flag,
            num_workers = self.num_workers,
            drop_last=self.drop_last
        )

    def val_dataloader(self):
        shuffle_flag = True
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=shuffle_flag,
            num_workers = self.num_workers,
            drop_last=self.drop_last
        )

    def test_dataloader(self):
        shuffle_flag = False
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=shuffle_flag,
            num_workers = self.num_workers,
            drop_last=self.drop_last
        )




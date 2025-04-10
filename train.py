from datetime import datetime
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from src.iterator import KlueStsDataSet
# from dataloader import KlueStsDataLoaderFetcher
from src.model import RoBertaModel, Summary
from src.progress_bar import ProgressBar

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class DataPath:
    REPO_ID = "klue/klue"
    LOCAL_DIR = Path("klue-sts-v1.1/hub")
    TRAIN_FILE = 'sts/train-00000-of-00001.parquet'
    TEST_FILE = 'sts/validation-00000-of-00001.parquet'

    @property
    def local_dir(self) -> str:
        return str(self.LOCAL_DIR)

    @property
    def files(self) -> list[str]:
        return [self.TRAIN_FILE, self.TEST_FILE]

    @property
    def local_train_file(self) -> Path:
        return self.LOCAL_DIR.joinpath(self.TRAIN_FILE)

    @property
    def local_test_file(self) -> Path:
        return self.LOCAL_DIR.joinpath(self.TEST_FILE)


class Trainer:
    def __init__(self):
        self.data_path = DataPath()
        self.model = RoBertaModel().to(DEVICE)
        self.epoch = 2
        self.log_file = Path('log.txt')

    def logger(self, message: str):
        print(message)
        self.log_file.open('+a').write(message + '\n')

    def run(self):
        train_loader = (
            KlueStsDataSet(tokenizer=self.model.tokenizer)
            .fetch(str(self.data_path.local_train_file))
            .get_dataloader(batch_size=16, shuffle=True)
        )
        test_loader = (
            KlueStsDataSet(tokenizer=self.model.tokenizer)
            .fetch(str(self.data_path.local_test_file))
            .get_dataloader(batch_size=16, shuffle=False)
        )

        summary = Summary()
        test_summary = Summary()
        self.logger('train start : ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' -' * 50)
        for e in range(self.epoch):
            for data in ProgressBar(train_loader):
                output, loss, labels = self.model(*data, train=True)
                summary.update(loss, output, labels)
            train_report = summary.flush()
            with torch.no_grad():
                for data in ProgressBar(test_loader, graph=False):
                    output, loss, labels = self.model(*data, train=False)
                    test_summary.update(loss, output, labels)
                report = test_summary.flush()
            self.logger(f'epoch: {e + 1:03d} ' + train_report + ' | test ' + report)

    def setup(self):
        for name in self.data_path.files:
            result = hf_hub_download(
                repo_id=self.data_path.REPO_ID,
                repo_type="dataset",
                filename=name,
                local_dir=self.data_path.local_dir,
            )
            print(result)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.run()

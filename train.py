from pathlib import Path

from huggingface_hub import hf_hub_download

from dataloader import KlueStsDataLoaderFetcher
from src.model import RoBertaModel, Summary
from src.progress_bar import ProgressBar


class DataPath:
    # TRAIN = 'klue-sts-v1.1/klue-sts-v1.1_train.json'
    # SAMPLE = 'klue-sts-v1.1/klue-sts-v1.1_dev_sample_10.json'
    # TEST = 'klue-sts-v1.1/klue-sts-v1.1_dev.json'
    LOCAL_DIR = Path("klue-sts-v1.1/hub")
    REPO = ['sts/train-00000-of-00001.parquet', 'sts/validation-00000-of-00001.parquet']

    @property
    def local_train_file(self) -> str:
        return str(self.LOCAL_DIR.joinpath(self.REPO[0]))

    @property
    def local_test_file(self) -> str:
        return str(self.LOCAL_DIR.joinpath(self.REPO[1]))


class Trainer:
    def __init__(self):
        self.data_path = DataPath()

        self.model = RoBertaModel()
        self.dataset = KlueStsDataLoaderFetcher(
            tokenizer=self.model.tokenizer,
        )
        self.epoch = 2

    def run(self):
        train_loader = self.dataset.get_dataloader(
            file_path=self.data_path.local_train_file,
            batch_size=2,
        )
        test_loader = self.dataset.get_dataloader(
            file_path=self.data_path.local_test_file,
            batch_size=2,
        )

        summary = Summary()
        test_summary = Summary()

        for e in range(self.epoch):
            for data in (bar := ProgressBar(train_loader)):
                bar.update(prefix=f'epoch : {e}')
                output, loss, labels = self.model(*data, train=True)
                summary.update(loss, output, labels)
            train_report = summary.flush()

            for data in (bar := ProgressBar(test_loader, graph=False)):
                output, loss, labels = self.model(*data, train=False)
                test_summary.update(loss, output, labels)
            report = test_summary.flush()
            print(train_report + ' | test ' + report)

    @staticmethod
    def setup():
        for name in DataPath.REPO:
            result = hf_hub_download(
                repo_id="klue/klue",
                repo_type="dataset",
                filename=name,
                local_dir="klue-sts-v1.1/hub",
            )
            print(result)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.run()

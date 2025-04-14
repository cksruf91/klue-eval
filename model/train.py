from datetime import datetime
from pathlib import Path

import torch

from common.data import KlueDataLoader
from common.progress_bar import ProgressBar
from model.src.iterator import KlueStsDataSet
from model.src.model import RoBertaModel, Summary

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class SemanticTextSimilarityTrainer:
    def __init__(self):
        self.data_loader = KlueDataLoader(sample=False)
        self.model = RoBertaModel().to(DEVICE)
        print(self.model)
        self.epoch = 100
        self.log_file = Path('log.txt')

    def logger(self, message: str):
        print(message)
        self.log_file.open('+a').write(message + '\n')

    def run(self):
        train_loader = (
            KlueStsDataSet(tokenizer=self.model.tokenizer)
            .fetch(self.data_loader.train_dict())
            .get_dataloader(batch_size=16, shuffle=True)
        )
        test_loader = (
            KlueStsDataSet(tokenizer=self.model.tokenizer)
            .fetch(self.data_loader.test_dict())
            .get_dataloader(batch_size=16, shuffle=False)
        )

        summary = Summary()
        test_summary = Summary()
        self.logger('train start : ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' -' * 50)
        for e in range(self.epoch):
            for data in ProgressBar(train_loader):
                output, loss, labels = self.model(**data, train=True)
                summary.update(loss, output, labels)
            train_report = summary.flush()
            with torch.no_grad():
                for data in ProgressBar(test_loader, graph=False):
                    output, loss, labels = self.model(**data, train=False)
                    test_summary.update(loss, output, labels)
                report = test_summary.flush()
            self.logger(f'epoch: {e + 1:03d} ' + train_report + ' | test ' + report)

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
        self.epoch = 100
        self.log_file = Path('log.txt')
        self.checkpoint = Path('model/checkpoint/model.zip')
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    def logger(self, message: str):
        print(message)
        self.log_file.open('+a').write(message + '\n')

    def run(self):
        train_loader = (
            KlueStsDataSet(tokenizer=self.model.tokenizer, train=True)
            .fetch(self.data_loader.train_dict())
            .get_dataloader(batch_size=64, shuffle=True, num_workers=0)
        )
        test_loader = (
            KlueStsDataSet(tokenizer=self.model.tokenizer)
            .fetch(self.data_loader.test_dict())
            .get_dataloader(batch_size=64, shuffle=False, num_workers=0)
        )

        summary = Summary()
        test_summary = Summary()
        best = 0
        self.logger('train start : ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' -' * 50)
        for e in range(self.epoch):
            self.model.train()
            for data, label in ProgressBar(train_loader):
                output, loss, labels = self.model(**data, labels=label, train=True)
                summary.update(loss, output, labels)
            train_report = summary.flush()

            with torch.no_grad():
                self.model.eval()
                for data, label in ProgressBar(test_loader, graph=False):
                    output, loss, labels = self.model(**data, labels=label, train=False)
                    test_summary.update(loss, output, labels)
                f1_score = test_summary.get_score()
                if f1_score > best:
                    best = f1_score
                    self.model.save(self.checkpoint)
                report = test_summary.flush()
            self.logger(f'epoch: {e + 1:03d} ' + train_report + ' | test ' + report)

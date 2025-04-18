from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoTokenizer, AutoConfig


class Summary:
    def __init__(self):
        self.epoch: float = 1
        self.loss: list[float] = []
        self.f1_score: float = 0
        self.pred: list[int] = []
        self.labels: list[int] = []

    def update(self, loss: torch.Tensor, output: torch.Tensor, labels: torch.Tensor):
        self.loss.append(loss.item())
        self.pred.extend(
            torch.gt(output, 2.5).reshape(-1).int().cpu().tolist()
        )
        self.labels.extend(
            torch.gt(labels, 2.5).reshape(-1).int().cpu().tolist()
        )
        # self.labels.extend(labels.cpu().tolist())

    def get_score(self):
        return f1_score(
            y_true=self.labels,
            y_pred=self.pred,
        )

    def flush(self):
        loss = np.mean(self.loss)
        f1 = self.get_score()
        report = f"loss: {loss: 1.3f}, f1_score: {f1: 1.3f}"
        self.epoch += 1
        self.loss = []
        self.pred = []
        self.labels = []
        return report


class RoBertaModel(nn.Module):
    # Korean-SRoBERTa†
    def __init__(self):
        super(RoBertaModel, self).__init__()
        self.repo = 'klue/roberta-small'
        self.model = AutoModel.from_pretrained(self.repo)
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo)
        self.config = AutoConfig.from_pretrained(self.repo)

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.config.hidden_size, 1, bias=False),
        )

        # self.loss = nn.BCELoss()
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)

    @torch.inference_mode()
    def embedding(self, text: str) -> list[float]:
        tokens = self.tokenizer(text, return_tensors='pt')
        last_hidden_state = self.model(**tokens).last_hidden_state
        return last_hidden_state.mean(dim=1).cpu().tolist()

    def _back_propagation(self, loss):
        loss.backward()  # 역전파를 통해 그라디언트 계산
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0, norm_type=2.0, )
        self.optimizer.step()  # 파라미터 업데이트
        self.optimizer.zero_grad()  # 기존 그라디언트 초기화

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, train=True):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output = self.classifier(outputs.pooler_output)
        loss = self.loss(output, labels)
        if train:
            self._back_propagation(loss)
        return output, loss, labels

    def save(self, save_dir: Path):
        torch.save(self, save_dir)

    def load(self, save_dir: Path, device: torch.device):
        self.load_state_dict(
            torch.load(save_dir, map_location=device)
        )

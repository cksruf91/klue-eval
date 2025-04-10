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
            torch.gt(output[:, 0], 0.5).int().cpu().tolist()
        )
        self.labels.extend(labels.cpu().tolist())

    def flush(self):
        loss = np.mean(self.loss)
        f1 = f1_score(
            y_true=self.labels,
            y_pred=self.pred,
            average='macro',
        )
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
        repo = 'klue/roberta-small'
        self.model = AutoModel.from_pretrained(repo)
        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        self.config = AutoConfig.from_pretrained(repo)

        self.model.frozen = True  # 모델 파라미터 고정

        self.classifier = nn.Sequential(
            torch.nn.GELU(),
            torch.nn.Linear(self.config.hidden_size, 2, bias=True),
        )

        self.loss = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5, weight_decay=1e-2)

    def _back_propagation(self, loss):
        loss.backward()  # 역전파를 통해 그라디언트 계산
        self.optimizer.step()  # 파라미터 업데이트
        self.optimizer.zero_grad()  # 기존 그라디언트 초기화

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, train=True):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_states = outputs.last_hidden_state  # 마지막 히든 스테이트
        pooled_output = hidden_states[:, 0, :]  # [CLS] 토큰의 출력
        output = self.classifier(pooled_output)
        loss = self.loss(output[:, 0], labels)
        if train:
            self._back_propagation(loss)
        return output, loss, labels

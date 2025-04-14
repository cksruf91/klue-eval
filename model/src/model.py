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
            torch.gt(output, 0.5).int().cpu().tolist()
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
        self.repo = 'klue/roberta-small'
        self.model = AutoModel.from_pretrained(self.repo)
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo)
        self.config = AutoConfig.from_pretrained(self.repo)

        self.classifier = nn.Sequential(
            torch.nn.Linear(self.config.hidden_size, 1, bias=False),
        )

        self.loss = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)

    @torch.inference_mode()
    def embedding(self, text: str) -> list[float]:
        tokens = self.tokenizer(text, return_tensors='pt')
        last_hidden_state = self.model(**tokens).last_hidden_state
        return last_hidden_state.mean(dim=1).cpu().tolist()

    def _back_propagation(self, loss):
        loss.backward()  # 역전파를 통해 그라디언트 계산
        self.optimizer.step()  # 파라미터 업데이트
        self.optimizer.zero_grad()  # 기존 그라디언트 초기화

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, train=True):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output = self.classifier(outputs.pooler_output)
        loss = self.loss(output, labels)
        if train:
            self._back_propagation(loss)
        return output, loss, labels

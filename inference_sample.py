import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

from dataloader import KlueStsDataLoaderFetcher


def load_model_and_type():
    """load model and model type from tar file pre-fetched from s3
    """
    name = "klue/roberta-small"
    model = AutoModelForSequenceClassification.from_pretrained(name)
    config = AutoConfig.from_pretrained(name)
    return model, config.model_type


@torch.no_grad()
def inference(data_dir, output_dir, args) -> None:
    # configure gpu
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model, model_type = load_model_and_type()
    model.to(device)
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-small")

    # get test_data_loader
    klue_sts_dataloader_fetcher = KlueStsDataLoaderFetcher(tokenizer, args.max_length)
    kwargs = (
        {"num_workers": num_gpus, "pin_memory": True}
        if torch.cuda.is_available()
        else {}
    )
    klue_sts_test_loader = klue_sts_dataloader_fetcher.get_dataloader(
        file_path=os.path.join(data_dir, args.test_filename),
        batch_size=args.batch_size,
        **kwargs,
    )

    # infer
    output_file = open(os.path.join(output_dir, args.output_filename), "w")
    for out in klue_sts_test_loader:
        input_ids, attention_mask, token_type_ids, labels = [o.to(device) for o in out]
        if model_type == 'roberta':
            output = model(input_ids, attention_mask=attention_mask)[0]
        else:
            output = model(
                input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
            )[0]

        preds = output.detach().cpu().numpy()

        for p in preds:
            score = p[0]
            output_file.write(f"{score}\n")

    output_file.close()


class Arguments:
    def __init__(self):
        self.max_length = 510
        self.batch_size = 32
        self.output_filename = "output.txt"
        self.test_filename = "klue-sts-v1.1_dev_sample_10.json"


if __name__ == "__main__":
    # args = Arguments()
    # data_dir = 'klue-sts-v1.1'
    # output_dir = 'output'
    # inference(data_dir, output_dir, args)
    import json

    print("Loading data...")
    data = json.load(open('klue-sts-v1.1/klue-sts-v1.1_train.json', 'r', encoding='utf-8'))
    print(len(data))

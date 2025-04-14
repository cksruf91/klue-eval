import argparse

from llm.eval import Evaluator as LLMEvaluator
from model.eval import Evaluator as ModelEvaluator
from model.train import SemanticTextSimilarityTrainer

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('task', type=int,
                      help='task to run: 1 for training, 2 for model evaluation, 3 for LLM evaluation')
    args = args.parse_args()
    if args.task == 1:
        trainer = SemanticTextSimilarityTrainer()
        trainer.run()
    elif args.task == 2:
        evaluator = ModelEvaluator()
        evaluator.run()
    elif args.task == 3:
        evaluator = LLMEvaluator()
        evaluator.run()

import time
from pathlib import Path

import numpy as np
import torch
import wandb

import sys

PROJECT_PATH = f"{Path(__file__).absolute().parent.parent.parent.parent}"
sys.path.insert(1, PROJECT_PATH)

from clickstream_experiment.source.tagnn.abx import calculate_abx_score
from clickstream_experiment.source.tagnn.utils import trans_to_cuda, trans_to_cpu


class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def forward(self, i, data):
        alias_inputs, A, items, mask, targets = data.get_slice(i)

        alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
        items = trans_to_cuda(torch.Tensor(items).long())
        A = trans_to_cuda(torch.Tensor(np.array(A)).float())
        mask = trans_to_cuda(torch.Tensor(mask).long())

        hidden = self.model(items, A)
        get = lambda i: hidden[i][alias_inputs[i]]
        seq_hidden = torch.stack(
            [get(i) for i in torch.arange(len(alias_inputs)).long()]
        )
        return targets, self.model.compute_scores(seq_hidden, mask)

    def save_results(self, data, results_path, id2item=None):
        with open(results_path, "w") as out_file:
            self.model.eval()
            slices = data.generate_batch(self.model.batch_size)
            with torch.no_grad():
                for i in slices:
                    targets, scores = self.forward(i, data)
                    sub_scores = scores.topk(20)[1]
                    sub_scores = trans_to_cpu(sub_scores).detach().numpy()
                    for idx, scores in zip(i, sub_scores):
                        if id2item is not None:
                            items = [id2item[item_id] for item_id in scores + 1]
                        else:
                            items = [str(item_id) for item_id in scores + 1]
                        items = " ".join(items)
                        out_file.write(f"{idx} [{items}]\n")

    def test(self, data, abx_tests_pdf=None):
        self.model.eval()
        hit, mrr = [], []
        slices = data.generate_batch(self.model.batch_size)
        with torch.no_grad():
            for i in slices:
                targets, scores = self.forward(i, data)
                sub_scores = scores.topk(20)[1]
                sub_scores = trans_to_cpu(sub_scores).detach().numpy()
                for score, target, mask in zip(sub_scores, targets, data.mask):
                    hit.append(np.isin(target - 1, score))
                    if len(np.where(score == target - 1)[0]) == 0:
                        mrr.append(0)
                    else:
                        mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))

        metrics = {
            "Recall@20": np.mean(hit) * 100,
            "MRR@20": np.mean(mrr) * 100,
        }

        if abx_tests_pdf is not None:
            embeddings = np.array(self.model.embedding.weight.cpu().detach())
            abx_scores = calculate_abx_score(embeddings, abx_tests_pdf)
            metrics.update(abx_scores)

        return metrics


class Trainer(ModelWrapper):
    def __init__(self, model, log_freq=5, model_dir=None):
        super().__init__(model)
        self.log_freq = log_freq
        self.model_dir = model_dir

        self.best_recall = 0
        self.best_mrr = 0
        self.epoch = 0

    def train_batch(self, train_data, batch):
        self.model.train()
        self.model.optimizer.zero_grad()
        targets, scores = self.forward(batch, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = self.model.loss_function(scores, targets - 1)
        loss.backward()
        self.model.optimizer.step()
        return loss.item()

    def step(self, train_data, test_data, scheduler_step=True, abx_tests_pdf=None):
        loss = []
        slices = train_data.generate_batch(self.model.batch_size)
        improved_flag = 0
        for i, j in zip(slices, np.arange(len(slices))):
            loss.append(self.train_batch(train_data, i))

            if j % int(len(slices) / self.log_freq + 1) == 0:
                metrics = self.test(test_data, abx_tests_pdf)
                metrics["loss"] = np.mean(loss)

                if self.best_recall < metrics["Recall@20"]:
                    self.best_recall = metrics["Recall@20"]
                    improved_flag = 1
                if self.best_mrr < metrics["MRR@20"]:
                    self.best_mrr = metrics["MRR@20"]
                    improved_flag = 1
                metrics.update(
                    {
                        "Recall_best": self.best_recall,
                        "MRR_best": self.best_mrr,
                        "epoch": self.epoch,
                    }
                )

                wandb.log(metrics)
                print(f"[{j}/{len(slices)}]")

        if scheduler_step:
            self.model.scheduler.step()

        self.epoch += 1
        return improved_flag

    def fit(
        self,
        train_data,
        test_data,
        epochs,
        patience,
        unfreeze_embeddings=None,
        abx_tests_pdf=None,
    ):
        start = time.time()
        bad_counter = 0

        while self.epoch < epochs:
            if unfreeze_embeddings is not None and self.epoch == unfreeze_embeddings:
                self.model.unfreeze_embeddings()
            print("-------------------------------------------------------")
            print("epoch: ", self.epoch)

            if unfreeze_embeddings is not None and self.epoch < unfreeze_embeddings:
                flag = self.step(
                    train_data,
                    test_data,
                    scheduler_step=False,
                    abx_tests_pdf=abx_tests_pdf,
                )
            else:
                flag = self.step(
                    train_data,
                    test_data,
                    scheduler_step=True,
                    abx_tests_pdf=abx_tests_pdf,
                )

            print(f"Flag: {flag}")

            print("Best Result:")
            print(
                "\tRecall@20:\t%.4f\tMMR@20:\t%.4f" % (self.best_recall, self.best_mrr)
            )
            bad_counter += 1 - flag

            if self.model_dir is not None:
                torch.save(
                    self.model.state_dict(),
                    Path(self.model_dir) / f"epoch_{self.epoch}.pth",
                )

            print(f"Bad counter: {bad_counter}/{patience}")
            if bad_counter >= patience:
                break
        print("-------------------------------------------------------")
        metrics = self.test(test_data, abx_tests_pdf)

        if self.best_recall < metrics["Recall@20"]:
            self.best_recall = metrics["Recall@20"]
        if self.best_mrr < metrics["MRR@20"]:
            self.best_mrr = metrics["MRR@20"]
        metrics.update(
            {
                "Recall_best": self.best_recall,
                "MRR_best": self.best_mrr,
                "epoch": self.epoch,
            }
        )
        print(f"Final metrics: \n{metrics}")
        end = time.time()
        print("Run time: %f s" % (end - start))

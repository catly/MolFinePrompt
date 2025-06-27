import argparse
import os

import numpy as np
import torch
import tqdm
from sklearn import metrics
from torch.utils.data.dataloader import DataLoader

from datasets.read_datasets import DATASET_PATHS
from utils import set_seed
from datasets.ddi_dataset import ddiDataset
from models.prom_ddi import promSmiles

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
def train_classifier(prompt_model, optimizer, dataset, num_task, config, device):

    train_dataloader = DataLoader(
        dataset.train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
    )

    prompt_model.train()
    i = -1
    classifier_losses = []
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    best_val_auc = 0
    for i in range(config.epochs):
        progress_bar = tqdm.tqdm(
            desc="epoch % 3d" % (i + 1)
        )
        epoch_classifier_losses = []

        promdata_rep_total, label_total = None, None
        for bid, batch in enumerate(train_dataloader):
            batch = list(zip(*batch))
            promdata_rep = prompt_model(batch[0], batch[1])
            label = torch.tensor(batch[2]).to(device)
            if promdata_rep_total is None:
                promdata_rep_total = promdata_rep
                label_total = label
            else:
                promdata_rep_total = torch.cat((promdata_rep_total, promdata_rep), dim=0)
                label_total = torch.cat((label_total, label), dim=0)
            loss = criterion(promdata_rep, label.float().unsqueeze(1)).mean()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            epoch_classifier_losses.append(loss.item())
            progress_bar.set_postfix(
                {"train loss": "{:.7f}".format(np.mean(epoch_classifier_losses[-50:]))}
            )
            progress_bar.update()
        progress_bar.close()
        progress_bar.write(
            f"epoch {i + 1} train loss {np.mean(epoch_classifier_losses)}"
        )
        classifier_losses.append(np.mean(epoch_classifier_losses))

        train_roc_auc, train_acc = do_compute_metrics(promdata_rep_total, label_total, num_task)
        val_roc_auc, val_acc = eval(prompt_model, dataset.valid_dataset, config.eval_batch_size, num_task)

        if val_roc_auc > best_val_auc:
            best_val_auc = val_roc_auc
            torch.save(prompt_model.state_dict(),  os.path.join(
            'best_state.pt'))

        test_roc_auc, test_acc = eval(prompt_model, dataset.test_dataset, config.eval_batch_size, num_task)

    prompt_model.load_state_dict(torch.load(os.path.join(
        'best_state.pt')))
    test_roc_auc = eval(prompt_model, dataset.test_dataset, config.eval_batch_size, num_task)
    return prompt_model, optimizer


def eval(model, dataset, batch_size, num_task):

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    with torch.no_grad():
        model.eval()
        promdata_rep_total, label_total = None, None
        for bid, batch in enumerate(dataloader):
            batch = list(zip(*batch))
            promdata_rep = prompt_model(batch[0], batch[1])
            label = torch.tensor(batch[2]).to(device)

            if promdata_rep_total is None:
                promdata_rep_total = promdata_rep
                label_total = label
            else:
                promdata_rep_total = torch.cat((promdata_rep_total, promdata_rep), dim=0)
                label_total = torch.cat((label_total, label), dim=0)
    roc_auc, acc = do_compute_metrics(promdata_rep_total, label_total, num_task)
    return roc_auc, acc

def do_compute_metrics(rep, label, num_task):
    predicted = torch.sigmoid(rep).detach().cpu().numpy()
    label = label.detach().cpu().numpy()

    roc_auc_scores = metrics.roc_auc_score(label, predicted)
    predicted_labels = (predicted >= 0.5).astype(int)
    acc = metrics.accuracy_score(label, predicted_labels)
    return roc_auc_scores, acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name",help="name of the experiment",default="ddi",type=str,)
    parser.add_argument("--dataset", help="name of the dataset", default="DDI", type=str)

    parser.add_argument("--epochs", help="number of epochs", default=30, type=int)
    parser.add_argument("--train_batch_size", help="train batch size", default=32, type=int)
    parser.add_argument("--eval_batch_size", help="eval batch size", default=32, type=int )
    parser.add_argument("--evaluate_only", help="directly evaluate on the" "dataset without any training", action="store_true", )
    parser.add_argument("--attr_dropout", help="add dropout to attributes", type=float, default=0.0,)
    parser.add_argument('--prelr', type=float, default=0.001, help='learning rate (default: 0.0001)  0.0005')
    parser.add_argument("--seed", help="seed value", default=42, type=int)
    parser.add_argument("--gradient_accumulation_steps", help="number of gradient accumulation steps", default=1,type=int)
    parser.add_argument('--classlr', type=float, default=0.0005, help='learning rate (default: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--split', type=str, default='scaffold', help='split dataset  splitting in [random, scaffold]')
    parser.add_argument('--task', type=str, default='zhangDDI', help='task name in ["zhangDDI", "ChChMiner", "DeepDDI", "DrugbankDDI"]')
    parser.add_argument('--pretrained_model_filepath', type=str, default='result/pretrain', help="pretrained model")
    parser.add_argument("--prefix_length", type=int, default=100, help="Defines the length for prompt tuning.")
    parser.add_argument("--text_init", type=bool, default=True, help="Whether to use text initialization for prompt tuning or not.")
    parser.add_argument("--num_transformer_submodules", type=int, default=1, help="Set to 1 to add the prompt only to the encoder. Set to 2 to add the prompt to both the encoder and decoder.")
    parser.add_argument("--prompt_tuning_init_text", type=str, default="", help="The text used for prompt tuning initialization.")

    config = parser.parse_args()
    # set the seed value
    set_seed(config.seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("training details")
    dataset_path = DATASET_PATHS[config.dataset]

    if config.task == 'zhangDDI':       num_tasks = 1
    elif config.task == 'ChChMiner':     num_tasks = 1
    elif config.task == 'DeepDDI':     num_tasks = 1
    elif config.task == 'DrugbankDDI':     num_tasks = 86
    else:  raise ValueError("Invalid dataset name.")

    dataset = ddiDataset(dataset_path, task=config.task, num_tasks=num_tasks)
    prompt_model = promSmiles(num_tasks, config.pretrained_model_file, device)
    optimizer = torch.optim.Adam(
        [
            {"params": list(prompt_model.ptmodel.parameters()), 'lr': config.prelr},
            {"params": list(prompt_model.gmodel.parameters()), 'lr': config.prelr},
            {"params": list(prompt_model.pred_linear.parameters()), 'lr': config.classlr},

        ],weight_decay=config.weight_decay
    )

    model, optimizer = train_classifier(prompt_model, optimizer, dataset, num_tasks, config, device)
    print("done!")
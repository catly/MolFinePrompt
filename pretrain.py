import argparse
import os
import yaml
from datetime import datetime
import numpy as np
import torch
import tqdm
import logging
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader

from datasets.pre_compositionDataset import preCompositionDataset
from datasets.read_datasets import DATASET_PATHS
from datasets.ret_datapro import retDatapro
from models.compositional_modules import get_model
from utils import set_seed

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

import torch.utils.data
from torch_geometric.data import Batch, Dataset

def train_model(model, optimizer, train_dataset, val_dataset,ret_dataset, config, device):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True
    )
    model.train()

    loss_fn = CrossEntropyLoss()

    train_losses = []
    graph, text, dataidx = [], [], []
    raw_text, raw_graph = [], []
    graph_acc_value, text_acc_value = [], []
    g2t_rec_max, t2g_rec_max = 0.85 ,0

    torch.autograd.set_detect_anomaly(True)
    val_best = 0
    for i in range(config.epochs):
        progress_bar = tqdm.tqdm(
            desc="epoch % 3d" % (i + 1)
        )
        epoch_train_losses = []
        g_correct, t_correct, total = 0,0,0

        for bid,batch in enumerate(train_dataloader):
            batch = list(zip(*batch))
            batch_text, batch_graph, batch_dataidx = batch[0], batch[1], batch[2]
            loss, text_features, graph_features = model(batch_text, batch_graph)
            loss = loss.to(device)

            if i == config.epochs -1 :
                graph += graph_features.tolist()
                text += text_features.tolist()
                dataidx += batch_dataidx

            g_probs = loss.softmax(dim=1)
            g_predictions = g_probs.argmax(dim=1)
            t_probs = loss.T.softmax(dim=1)
            t_predictions = t_probs.argmax(dim=1)
            labels = torch.arange(loss.shape[0]).to(device)

            loss_text = loss_fn(loss.T.to(device), torch.arange(loss.shape[1]).to(device))
            loss_graph = loss_fn(loss, torch.arange(loss.shape[0]).to(device))
            loss = (loss_text + loss_graph) / 2
            loss.backward()

            g_correct = g_correct + (g_predictions == labels).sum().item()
            t_correct = t_correct + (t_predictions == labels).sum().item()
            total = total + labels.size(0)

            if ((bid + 1) % config.gradient_accumulation_steps == 0) or \
                    (bid + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()

            epoch_train_losses.append(loss.item())
            progress_bar.set_postfix(
                {"train loss":  "{:.7f}".format(np.mean(epoch_train_losses[-50:]))}
            )

            progress_bar.update()

        progress_bar.close()
        progress_bar.write(
            f"epoch {i + 1} train loss {np.mean(epoch_train_losses)}"
        )
        train_losses.append(np.mean(epoch_train_losses))

        g_accuracy = g_correct / total
        t_accuracy = t_correct / total

        graph_acc_value.append(g_accuracy)
        text_acc_value.append(t_accuracy)
        #val
        g2t_acc_v, t2g_acc_v = eval(val_dataset, model, config.train_batch_size)

        if g2t_acc_v > val_best:
            val_best = g2t_acc_v
            torch.save(model.state_dict(), os.path.join(f'model_best_state.pt'))

        g2t_acc, t2g_acc, g2t_rec, t2g_rec = eval_ret(ret_dataset, model, config.train_batch_size)
        if g2t_rec > g2t_rec_max :
            torch.save(model.state_dict(), os.path.join(f'model_state_{i+1}.pt'))

    save_embeddings(dataidx, text, graph, raw_text, raw_graph)

    return model, optimizer

def eval(dataset, model, batch_size):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )
    losses = []
    loss_fn = CrossEntropyLoss()
    with torch.no_grad():
        model.eval()
        g2t_acc, t2g_acc, allnum = 0, 0, 0
        for bid, batch in enumerate(dataloader):
            batch = list(zip(*batch))
            batch_text, batch_graph, batch_dataidx = batch[0], batch[1], batch[2]
            loss, text_rep, graph_rep = model(batch_text, batch_graph)
            loss_text = loss_fn(loss.T.to(device), torch.arange(loss.shape[1]).to(device))
            loss_graph = loss_fn(loss, torch.arange(loss.shape[0]).to(device))
            loss = (loss_text + loss_graph) / 2
            losses.append(loss.item())

            g2t_sim = torch.cosine_similarity(
                graph_rep.unsqueeze(1).expand(graph_rep.shape[0], graph_rep.shape[0], graph_rep.shape[1]),
                text_rep.unsqueeze(0).expand(text_rep.shape[0], text_rep.shape[0], text_rep.shape[1]), dim=-1)
            t2g_sim = torch.cosine_similarity(
                text_rep.unsqueeze(1).expand(text_rep.shape[0], text_rep.shape[0], text_rep.shape[1]),
                graph_rep.unsqueeze(0).expand(graph_rep.shape[0], graph_rep.shape[0], graph_rep.shape[1]), dim=-1)

            g2t_argm = torch.argmax(g2t_sim, dim=1).to(device)
            t2g_argm = torch.argmax(t2g_sim, dim=1).to(device)

            g2t_acc += sum((g2t_argm == torch.arange(g2t_argm.shape[0]).to(device)).int()).item()
            t2g_acc += sum((t2g_argm == torch.arange(t2g_argm.shape[0]).to(device)).int()).item()

            allnum += g2t_argm.shape[0]
    return g2t_acc / allnum, t2g_acc / allnum

def eval_ret(dataset, model, batch_size):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )
    with torch.no_grad():
        model.eval()
        g2t_acc, t2g_acc, allnum = 0, 0, 0
        graph_rep_total, text_rep_total = None, None
        for bid, batch in enumerate(dataloader):
            batch = list(zip(*batch))
            batch_text, batch_graph, batch_dataidx = batch[0], batch[1], batch[2]
            _, text_rep, graph_rep = model(batch_text, batch_graph)

            g2t_sim = torch.cosine_similarity(
                graph_rep.unsqueeze(1).expand(graph_rep.shape[0], graph_rep.shape[0], graph_rep.shape[1]),
                text_rep.unsqueeze(0).expand(text_rep.shape[0], text_rep.shape[0], text_rep.shape[1]), dim=-1)
            t2g_sim = torch.cosine_similarity(
                text_rep.unsqueeze(1).expand(text_rep.shape[0], text_rep.shape[0], text_rep.shape[1]),
                graph_rep.unsqueeze(0).expand(graph_rep.shape[0], graph_rep.shape[0], graph_rep.shape[1]), dim=-1)

            g2t_argm = torch.argmax(g2t_sim, dim=1).to(device)
            t2g_argm = torch.argmax(t2g_sim, dim=1).to(device)

            g2t_acc += sum((g2t_argm == torch.arange(g2t_argm.shape[0]).to(device)).int()).item()
            t2g_acc += sum((t2g_argm == torch.arange(t2g_argm.shape[0]).to(device)).int()).item()

            allnum += g2t_argm.shape[0]

            if graph_rep_total is None or text_rep_total is None:
                graph_rep_total = graph_rep
                text_rep_total = text_rep
            else:
                graph_rep_total = torch.cat((graph_rep_total, graph_rep), dim=0)
                text_rep_total = torch.cat((text_rep_total, text_rep), dim=0)

        g2t_rec = rec20(graph_rep_total, text_rep_total)
        t2g_rec = rec20(text_rep_total, graph_rep_total)
    return g2t_acc / allnum, t2g_acc / allnum, g2t_rec, t2g_rec

def rec20(rep1, rep2):
    sim = torch.zeros(rep1.shape[0], rep1.shape[0])
    rep2T = rep2.t()
    for i in range(rep1.size(0)):
        sim[i] = torch.matmul(rep1[i], rep2T)

    rec_idx = []
    for i in range(sim.size(0)):
        sorted_scores, sorted_indices = torch.sort(sim[i], descending=True)
        for j in range(sorted_indices.size(0)):
            if sorted_indices[j] == i:
                rec_idx.append(j)
                break

    rec = sum((np.array(rec_idx) < 20).astype(int)) / sim.size(0)
    return rec

def save_embeddings(dataidx, text, graph, raw_text, raw_graph):
    data = {'dataidx' : dataidx,
            'textfeatures': text,
            'graphfeatures': graph,
            'rawtext' : raw_text,
            'rawgraph' : raw_graph
            }

    save_text_emb = os.path.join("emb.pth")
    torch.save(data, save_text_emb)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", help="name of the experiment", default="drugprompt", type=str,)
    parser.add_argument("--dataset", help="name of the dataset", default="pubchem", type=str)
    parser.add_argument("--retdataset", help="name of the dataset [pubchem_test, kv_data, phy_data, ChEBI-20_data]",default="kv_data", type=str)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-05)
    parser.add_argument("--weight_decay", help="weight decay", type=float, default=0)  #1e-02
    parser.add_argument("--epochs", help="number of epochs", default=100, type=int)
    parser.add_argument("--train_batch_size", help="train batch size", default= 32, type=int)
    parser.add_argument("--eval_batch_size", help="eval batch size", default= 32, type=int)
    parser.add_argument("--save_path", help="save path", default='./models',type=str)
    parser.add_argument("--save_every_n", default=1, type=int, help="saves the model every n epochs; ")
    parser.add_argument("--save_model", help="indicate if you want to save the model state dict()", action="store_true",)
    parser.add_argument("--seed", help="seed value", default=42, type=int)
    parser.add_argument("--gradient_accumulation_steps", help="number of gradient accumulation steps", default=1, type=int)
    parser.add_argument("--logit_scale_lr", default= 1e-5, type=float)
    parser.add_argument("--text_encoder_state", default= "fine", type=str)

    parser.add_argument('--glr', type=float, default=0.0001, help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0, help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=768, help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last", help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--graph_pretrain_file', type=str, default='graph/saved_model/pretrain.pth',help='filename to read the model (if there is any)')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataset loading')
    parser.add_argument("--hidden_size", type=int, default=768, help='hidden size')
    parser.add_argument("--fine_tune_from", type=str, default="161w pretrained")

    config = parser.parse_args()
    # set the seed value
    set_seed(config.seed)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print("training details")

    dataset_path = DATASET_PATHS[config.dataset]
    train_dataset = preCompositionDataset(dataset_path,
                                       phase='train',
                                       type = 'pertrain')
    val_dataset = preCompositionDataset(dataset_path,
                                       phase='val',
                                       type = 'pertrain')

    ret_dataset = retDatapro(data, dataset=config.retdataset, type="test")
    model, optimizer = get_model(config, device)
    model_state = torch.load('result/pretrain/model_best_state.pt')

    model.load_state_dict(model_state)
    model, optimizer = train_model(
        model,
        optimizer,
        train_dataset,
        val_dataset,
        ret_dataset,
        config,
        device
    )

    torch.save(model, os.path.join('final_model.pt'))
    print("done!")
import argparse
import os
from datetime import datetime
import numpy as np
import torch
import tqdm
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from datasets.ret_datapro import retDatapro
from datasets.read_datasets import DATASET_PATHS
from utils import set_seed

import torch.utils.data
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def train_model(model, optimizer, train_dataset, vaild_dataset, test_dataset,config, device):

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True
    )

    model.train()

    loss_fn = CrossEntropyLoss()
    i = 0
    train_losses = []
    graph, text, dataidx, atc = [], [], [], []
    train_acc_list, val_g2t_acclist, val_t2g_acclist, val_g2t_reclist, val_t2g_reclist = [], [], [], [], []
    best_val_acc = -1
    torch.autograd.set_detect_anomaly(True)

    for i in range(config.epochs):
        progress_bar = tqdm.tqdm(
            desc="epoch % 3d" % (i + 1)
        )

        epoch_train_losses = []
        g_correct, t_correct, total = 0,0,0
        for bid,batch in enumerate(train_dataloader):
            batch = list(zip(*batch))
            batch_text, batch_graph = batch[0] , batch[1]
            loss, text_features, graph_features = model(batch_text, batch_graph)
            loss = loss.to(device)

            if i == config.epochs -1 :
                graph += graph_features.tolist()
                text += text_features.tolist()

            g_probs = loss.softmax(dim=1)
            g_predictions = g_probs.argmax(dim=1)
            t_probs = loss.T.softmax(dim=1)
            t_predictions = t_probs.argmax(dim=1)

            labels = torch.arange(loss.shape[0]).to(device)
            loss_text = loss_fn(loss.T.to(device), torch.arange(loss.shape[1]).to(device))
            loss_graph = loss_fn(loss, torch.arange(loss.shape[0]).to(device))
            loss = (loss_text + loss_graph)/2
            loss.backward()

            g_correct += (g_predictions == labels).sum().item()
            t_correct += (t_predictions == labels).sum().item()
            total += labels.size(0)

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
        total_acc = (g_accuracy+t_accuracy)/2
        train_acc_list.append(total_acc)

        val_g2t_acc, val_t2g_acc, val_g2t_rec, val_t2g_rec = eval(vaild_dataset, model, config.train_batch_size)

        val_g2t_acclist.append(val_g2t_acc)
        val_t2g_acclist.append(val_t2g_acc)
        val_g2t_reclist.append(val_g2t_rec)
        val_t2g_reclist.append(val_t2g_rec)
        val_acc = (val_g2t_acc + val_t2g_acc)/2

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_model.pt")

        ttest_g2t_acc, ttest_t2g_acc, ttest_g2t_rec, ttest_t2g_rec = eval(test_dataset, model, config.test_batch_size)

    best_model_path = f"best_model.pt"
    model.load_state_dict(torch.load(best_model_path))
    test_g2t_acc, test_t2g_acc, test_g2t_rec, test_t2g_rec = eval(test_dataloader, model, config.test_batch_size)
    save_embeddings(dataidx, text, graph)

    return model, optimizer

def eval(dataset, model, batch_size):

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )
    with torch.no_grad():
        model.eval()
        g2t_acc, t2g_acc, allnum = 0, 0, 0
        g2t_rec20, t2g_rec20 = 0, 0
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

            g2t_rank = (g2t_sim.argsort(descending=True) == torch.arange(g2t_argm.shape[0]).reshape(-1, 1).to(device)).int().argmax(dim=1)
            t2g_rank = (t2g_sim.argsort(descending=True) == torch.arange(g2t_argm.shape[0]).reshape(-1, 1).to(device)).int().argmax(dim=1)

            g2t_rec20 += float((g2t_rank < 20).sum())
            t2g_rec20 += float((t2g_rank < 20).sum())

            allnum += g2t_argm.shape[0]

            if graph_rep_total is None or text_rep_total is None:
                graph_rep_total = graph_rep
                text_rep_total = text_rep
            else:
                graph_rep_total = torch.cat((graph_rep_total, graph_rep), dim=0)
                text_rep_total = torch.cat((text_rep_total, text_rep), dim=0)

        g2t_all_acc = all_acc(graph_rep_total, text_rep_total)
        t2g_all_acc = all_acc(text_rep_total, graph_rep_total)
        g2t_rec = rec20(graph_rep_total, text_rep_total)

        t2g_rec = rec20(text_rep_total, graph_rep_total)
        g2t_rec20_batch = round(g2t_rec20 / allnum * 100, 2)
        t2g_rec20_batch = round(t2g_rec20 / allnum * 100, 2)
    return g2t_all_acc, t2g_all_acc, g2t_acc/allnum, t2g_acc/allnum, g2t_rec, t2g_rec, g2t_rec20_batch,t2g_rec20_batch

def all_acc(rep1, rep2):
    device = rep1.device
    sim = torch.cosine_similarity(rep1.unsqueeze(1), rep2.unsqueeze(0), dim=-1)
    argm = torch.argmax(sim, dim=1).to(device)

    all_acc = sum((argm == torch.arange(argm.shape[0]).to(device)).int()).item()
    all_allnum = argm.shape[0]
    return all_acc / all_allnum

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

def save_embeddings(dataidx, text, graph):
    data = {'dataidx' : dataidx,
            'textfeatures': text,
            'graphfeatures': graph }

    save_text_emb = os.path.join("emb.pth")
    torch.save(data, save_text_emb)

class Config:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

    def __getattr__(self, name):
        return self.__dict__.get(name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "--experiment_name", help="name of the experiment", default="retrieval", type=str)
    parser.add_argument("--dataset", help="name of the dataset [pubchem_test, kv_data, phy_data]", default="pubchem_test", type=str)
    parser.add_argument("--weight_decay", help="weight decay", type=float, default=0 )
    parser.add_argument("--epochs", help="number of epochs", default=30, type=int)
    parser.add_argument("--train_batch_size", help="train batch size", default= 32 , type=int)
    parser.add_argument("--test_batch_size", help="eval batch size", default=32, type=int)
    parser.add_argument("--evaluate_only",help="directly evaluate on the" "dataset without any training",action="store_true",)
    parser.add_argument("--save_every_n",default=1,type=int,help="saves the model every n epochs; this is useful for validation/grid search",)
    parser.add_argument("--save_model",help="indicate if you want to save the model state dict()",action="store_true")
    parser.add_argument("--seed", help="seed value", default=42, type=int)
    parser.add_argument("--gradient_accumulation_steps",help="number of gradient accumulation steps",default=1,type=int)
    parser.add_argument('--num_workers', type=int, default=3, help='number of workers for dataset loading')

    parser.add_argument("--logit_scale_lr", default=1e-04, type=int)
    parser.add_argument('--glr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-05)
    parser.add_argument('--pretrained_model_file', type=str,
                        default='result/pretrain',
                        help="pretrained model")
    parser.add_argument("--mode", type=str, default="zeroshot", help="[finetune, zeroshot]")

    config = parser.parse_args()
    set_seed(config.seed)

    device = "cuda:0"
    print("training details")
    dataset_path = DATASET_PATHS[config.dataset]

    model = torch.load(os.path.join(config.pretrained_model_file, 'final_model.pt'))
    model_state = torch.load(os.path.join('result/pretrain/model_state.pt'))
    model.load_state_dict(model_state)
    print("model state done!")

    if config.mode == "finetune":
        train_dataset = retDatapro(dataset_path, dataset=config.dataset, type="train")
        vaild_dataset = retDatapro(dataset_path, dataset=config.dataset, type="vaild")
        test_dataset = retDatapro(dataset_path, dataset=config.dataset, type="test")
        optimizer = torch.optim.Adam(
            [
                {"params": model.logit_scale, 'lr': config.logit_scale_lr},
                {"params": model.text_encoder.parameters(), 'lr': config.lr},
                {"params": model.graph_encoder.model.gnn.parameters(), 'lr': config.glr}
            ], weight_decay=config.weight_decay
        )
        model, optimizer = train_model(
            model,
            optimizer,
            train_dataset,
            vaild_dataset,
            test_dataset,
            config,
            device,
        )
        torch.save(
            model.state_dict(),
            os.path.join(
                'final_model.pt'))
    else:
        for param in model.parameters():
            param.requires_grad = False
        test_dataset = retDatapro(dataset_path, dataset=config.dataset, type="test")
        g2t_all_acc, t2g_all_acc ,g2t_acc, t2g_acc, g2t_rec, t2g_rec, g2t_rec20_batch,t2g_rec20_batch = eval(test_dataset, model, config.train_batch_size)

    print("done!")



import torch
from torch import nn
from src import model as mm
from src.utils import *
import torch.optim as optim
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau


from src.eval_metrics import *


def initiate(hyp_params, train_loader, valid_loader, test_loader):
    if hyp_params.pretrained_model is not None:
        model = getattr(mm, "PromptModel")(hyp_params)
        model = transfer_model(model, hyp_params.pretrained_model)
    else:
        model = getattr(mm, "MULTModel")(hyp_params)

    if hyp_params.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=hyp_params.when, factor=0.1, verbose=True
    )
    settings = {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "scheduler": scheduler,
    }
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings["model"]
    optimizer = settings["optimizer"]
    criterion = settings["criterion"]
    scheduler = settings["scheduler"]

    def train(model, optimizer, criterion):
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        for i_batch, (batch_X, batch_Y, missing_mod) in enumerate(train_loader):
            text, audio, vision = batch_X
            eval_attr = batch_Y.squeeze(-1)
            model.zero_grad()

            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr = (
                        text.cuda(),
                        audio.cuda(),
                        vision.cuda(),
                        eval_attr.cuda(),
                    )
                    if hyp_params.dataset == "iemocap":
                        eval_attr = eval_attr.long()

            batch_size = text.size(0)
            net = nn.DataParallel(model) if batch_size > 10 else model
            preds = net(text, audio, vision, missing_mod)

            if hyp_params.dataset == "iemocap":
                preds = preds.view(-1, 4)
                eval_attr = eval_attr.view(-1)
            raw_loss = criterion(preds, eval_attr)
            raw_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print(
                    "Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}".format(
                        epoch,
                        i_batch,
                        num_batches,
                        elapsed_time * 1000 / hyp_params.log_interval,
                        avg_loss,
                    )
                )
                proc_loss, proc_size = 0, 0
                start_time = time.time()

    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0
        results = []
        truths = []

        with torch.no_grad():
            for i_batch, (batch_X, batch_Y, missing_mod) in enumerate(loader):
                text, audio, vision = batch_X
                eval_attr = batch_Y.squeeze(dim=-1)  # if num of labels is 1

                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr = (
                            text.cuda(),
                            audio.cuda(),
                            vision.cuda(),
                            eval_attr.cuda(),
                        )
                        if hyp_params.dataset == "iemocap":
                            eval_attr = eval_attr.long()

                batch_size = text.size(0)
                net = nn.DataParallel(model) if batch_size > 10 else model
                preds = net(text, audio, vision, missing_mod)
                if hyp_params.dataset == "iemocap":
                    preds = preds.view(-1, 4)
                    eval_attr = eval_attr.view(-1)
                total_loss += criterion(preds, eval_attr).item() * batch_size

                results.append(preds)
                truths.append(eval_attr)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    best_valid = 1e8
    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()
        train(model, optimizer, criterion)
        val_loss, _, _ = evaluate(model, criterion, test=False)
        test_loss, _, _ = evaluate(model, criterion, test=True)

        end = time.time()
        duration = end - start
        scheduler.step(val_loss)

        print("-" * 50)
        print(
            "Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}".format(
                epoch, duration, val_loss, test_loss
            )
        )
        print("-" * 50)

        if val_loss < best_valid:
            print(f"Saved model at {hyp_params.name}")
            torch.save(model, hyp_params.name)
            best_valid = val_loss

    model = torch.load(hyp_params.name)
    _, results, truths = evaluate(model, criterion, test=False)

    if hyp_params.dataset == "mosei":
        eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == "mosi":
        eval_mosi(results, truths, True)
    elif hyp_params.dataset == "iemocap":
        eval_iemocap(results, truths)
    elif hyp_params.dataset == "sims":
        eval_sims(results, truths)

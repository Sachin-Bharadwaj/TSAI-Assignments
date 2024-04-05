import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_epoch(model, device, train_loader, optimizer, criterion, epoch, train_stats):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        y_pred = model(data)

        # Calculate loss
        #loss = F.nll_loss(y_pred, target)
        loss = criterion(y_pred, target)

        if "loss" in train_stats:
            train_stats["loss"].append(loss)
        else:
            train_stats["loss"] = [loss]
        # train_losses.append(loss)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update pbar-tqdm

        pred = y_pred.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(
            desc=f"Loss={loss.item()} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}"
        )
        if "acc" in train_stats:
            train_stats["acc"].append(100 * correct / processed)
        else:
            train_stats["acc"] = [100 * correct / processed]
        # train_acc.append(100 * correct / processed)


def test_epoch(model, device, test_loader, criterion, test_stats):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #test_loss += F.nll_loss(
            #    output, target, reduction="sum"
            #).item()  # sum up batch loss
            test_loss += criterion(output, target).item()
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    if "loss" in test_stats:
        test_stats["loss"].append(test_loss)
    else:
        test_stats["loss"] = [test_loss]
    # test_losses.append(test_loss)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    if "acc" in test_stats:
        test_stats["acc"].append(100.0 * correct / len(test_loader.dataset))
    else:
        test_stats["acc"] = [100.0 * correct / len(test_loader.dataset)]
    # test_acc.append(100. * correct / len(test_loader.dataset))

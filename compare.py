import torch

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST

from torchvision import transforms
from tqdm import tqdm


def unitarize(A: torch.tensor):
    """
    unitarize a matrix A
    """

    L, V = torch.linalg.eigh(torch.bmm(A.transpose(1, 2), A))
    # rint(L, V)
    return A @ V @ torch.diag_embed((L + 1e-5) ** -0.5) @ V.transpose(1, 2)


class UnnLSTM(nn.Module):
    def __init__(self, uni=True, *args, **kwargs):
        super(UnnLSTM, self).__init__()
        self.lstm = nn.LSTM(*args, **kwargs)
        self.uni = uni
        self.mlp = nn.Sequential(nn.ReLU(), nn.Linear(self.lstm.hidden_size, 10))
        for name, layer in self.lstm.named_parameters():
            print(name)

    def forward(self, *args, **kwargs):

        if self.uni:
            with torch.no_grad():
                w_hi, w_hf, w_hc, w_ho = self.lstm.weight_hh_l0.chunk(4, 0)

                d = w_hi.shape[0]
                # print(w_hi.shape)
                w_hi, w_hf, w_hc, w_ho = (
                    w_hi.unsqueeze(0),
                    w_hf.unsqueeze(0),
                    w_hc.unsqueeze(0),
                    w_ho.unsqueeze(0),
                )
                w_hi = unitarize(w_hi).squeeze()
                w_hf = unitarize(w_hf).squeeze()
                w_hc = unitarize(w_hc).squeeze()
                w_ho = unitarize(w_ho).squeeze()

                x = torch.cat([w_hi, w_hf, w_hc, w_ho], 0)

                # To see the effect of projection,
                # print(nn.MSELoss()(x, self.lstm.weight_hh_l0))

                self.lstm.load_state_dict({"weight_hh_l0": x}, strict=False)

            self.lstm.flatten_parameters()

        _, (h, c) = self.lstm(*args, **kwargs)
        return self.mlp(c[0])


def train_mnist(n_epoch: int = 100, device="cuda:0", uni=True) -> None:

    model = UnnLSTM(input_size=1, hidden_size=64, batch_first=True, uni=uni).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=4e-3)

    # basic lstm experiments.
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(7),
            transforms.Normalize((0.5,), (1.0)),
        ]
    )
    trainset = MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )
    valset = MNIST("./data", train=False, download=True, transform=tf)
    train_dl = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    val_dl = DataLoader(valset, batch_size=128, shuffle=True, num_workers=4)

    for i in range(n_epoch):
        model.train()

        pbar = tqdm(train_dl)
        loss_ema = None
        for x, y in pbar:
            optim.zero_grad()
            x = x.to(device)
            y = y.to(device)
            x = x.flatten(1).unsqueeze(2)
            yh = model(x)

            loss = nn.CrossEntropyLoss()(yh, y)
            loss.backward()

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        model.eval()
        with torch.no_grad():
            corrects = 0

            for x, y in val_dl:
                x = x.to(device)
                y = y.to(device)
                x = x.flatten(1).unsqueeze(2)
                yh = model(x)
                corrects += (yh.argmax(dim=1) == y).sum().item()

            print(f"epoch {i}: {corrects / len(valset):.4f}")


if __name__ == "__main__":
    train_mnist()

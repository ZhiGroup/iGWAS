from QC_dataset import QCDataset
from model import QAModel
from torch.utils.data import DataLoader
from torch.nn import BCELoss
from torch.optim import Adam as OPT
import torch


device = "cuda"
n_epoch = 10
model = QAModel().to(device)
loader = DataLoader(QCDataset(), batch_size=32)
opt = OPT(model.parameters(), lr=1e-4)
loss = BCELoss()
for _ in range(n_epoch):
    loss_total = 0
    for data, label in loader:
        data = data.to(device)
        label = label.to(device)
        opt.zero_grad()
        l = loss(model(data), label.view(-1, 1).float())
        with torch.no_grad():
            loss_total += l
        l.backward()
        opt.step()
    print(loss_total/1001)
torch.save(model.state_dict(), "QC_model_wt.pth")
        
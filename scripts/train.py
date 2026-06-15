import torch
from monai.networks.nets import UNet
from monai.losses import DiceLoss

print("STARTING TRAINING")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2,2,2,2),
).to(device)

loss_function = DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# fake data (temporary OK)
x = torch.rand((1,1,64,64,64)).to(device)
y = torch.rand((1,1,64,64,64)).to(device)

model.train()

for epoch in range(5):
    optimizer.zero_grad()

    pred = model(x)
    loss = loss_function(pred, y)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1} | Loss: {loss.item()}")

torch.save(model.state_dict(), "model.pth")

print("MODEL SAVED")
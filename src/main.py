import torch
import numpy
import data_loader
from torchvision import models
import torch.nn as nn

#hyperparameters
BATCH_SIZE = 5
N_EPOCHS = 50
LR = 0.05
NUM_CLASSES = 14

image_loader_train, image_loader_validation = data_loader.load_data(BATCH_SIZE)

class MultilabelClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet34(pretrained=True)
        # MUUTA NIIN, ETTÄ ENSIMMÄINEN LAYER OTTAA GREYSCALE IMAGEN (1 CHANNEL 3 SIJAAN)
        # RuntimeError: The size of tensor a (5) must match the size of tensor b (14) at non-singleton dimension 1
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.resnet.fc.in_features, out_features=NUM_CLASSES)
            )
        self.model = self.resnet
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.sigm(x)
        return x
    
class EarlyStopper:
    def __init__(self, patience=1):
        self.patience = patience
        self.wait = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print("Early stopping due to improvement halt")
                return True
        return False
    
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = MultilabelClassifier()
model.to(device)

# tai Adam
optimizer = torch.optim.Adagrad(model.parameters(), lr=LR)
# tai CrossEntropyLoss
# katso https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/
# https://towardsdatascience.com/multilabel-classification-with-pytorch-in-5-minutes-a4fa8993cbc7
loss_function = nn.BCELoss()

# Training
early_stopper = EarlyStopper(patience=3)
model.train()
for epoch in range(N_EPOCHS):
    train_loss = 0
    train_correct = 0
    total = 0
    for batch_number, data in enumerate(image_loader_train):
        images, target_labels = data['image'].to(device), data['target_labels'].to(device)

        outputs = model(images)
        loss = loss_function(outputs, target_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        total += target_labels.size(0)
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == target_labels).sum().item()

        print('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' % 
              (epoch, batch_number, len(image_loader_train), train_loss / (batch_number + 1), 
               100. * train_correct / total, train_correct, total))
        
'''dataiter = iter(image_loader_train)
print(len(image_loader_train))
mini_batch = next(dataiter)

print(mini_batch["image"].shape)
print(mini_batch["target_labels"].shape)
'''
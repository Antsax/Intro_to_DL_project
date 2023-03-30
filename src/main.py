import torch
import numpy as np
import data_loader
from torchvision import models
import torch.nn as nn

#hyperparameters
BATCH_SIZE = 5
N_EPOCHS = 2
LR = 0.05
NUM_CLASSES = 14

image_loader_train, image_loader_validation = data_loader.load_data(BATCH_SIZE)

class MultilabelClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet34(weights='DEFAULT')
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

#HELPER FUNCTIONS

#turn the sigmoid probabilities to a torch tensor of zeros and ones. One if greater than the specified threshold
def output_to_prediction(outputs, threshold=0.5):
    predictions = []
    for output in outputs:
        prediction = [np.float32(1.0) if i>threshold else np.float32(0.0) for i in output]
        predictions.append(prediction)
    return torch.tensor(predictions).to(device)

#checks if all of the 14 labels are predicted correctly for one image (return True/False)
def prediction_fully_correct(prediction, target_label):
    return torch.equal(prediction, target_label)

#checks how many of the labels are predicted correctly for one image (returns int between 0 and 14)
def correct_labels_in_prediction(prediction, target_labels):
    return (prediction == target_labels).sum().item()

# Training
early_stopper = EarlyStopper(patience=3)
model.train()
for epoch in range(N_EPOCHS):
    train_loss = 0
    train_correct = 0
    total = 0
    total_correct_labels = 0
    for batch_number, data in enumerate(image_loader_train):
        images, target_labels = data['image'].to(device), data['target_labels'].to(device)

        outputs = model(images)
        loss = loss_function(outputs, target_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        total += target_labels.size(0)
        predictions = output_to_prediction(outputs)
        for i, prediction in enumerate(predictions):
            if prediction_fully_correct(prediction, target_labels[i]):
                train_correct += 1
            total_correct_labels += correct_labels_in_prediction(prediction, target_labels[i])

        print('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train fully correct: %.3f%% (%d/%d) - Train labels correct: %.3f%% (%d/%d)' % 
              (epoch, batch_number + 1, len(image_loader_train), train_loss / (batch_number + 1), 
               100. * train_correct / total, train_correct, total, 100. * total_correct_labels / (total*14), total_correct_labels, (total*14)))
        
    #validation after each epoch
    validation_loss = 0
    with torch.no_grad():
        for batch_number, data in enumerate(image_loader_validation):
            images, target_labels = data['image'].to(device), data['target_labels'].to(device)
            
            outputs = model(images)
            
            loss = loss_function(outputs, target_labels)
            validation_loss += loss.item()
            
    if early_stopper.early_stop(validation_loss):
        break
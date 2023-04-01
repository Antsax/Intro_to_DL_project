import torch
import numpy as np
import data_loader
from torchvision import models
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
import csv
import os

#hyperparameters
BATCH_SIZE = 50
N_EPOCHS = 2
LR = 0.1
NUM_CLASSES = 14

image_loader_train, image_loader_validation = data_loader.load_data(BATCH_SIZE)
image_loader_test = data_loader.load_test_data(BATCH_SIZE)

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

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
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
for epoch in range(N_EPOCHS):
    train_loss = 0
    train_correct = 0
    total = 0
    total_correct_labels = 0
    model.train()
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
    print("validating")
    validation_loss = 0
    model.eval()
    with torch.no_grad():
        all_predictions = []
        all_true_labels = []
        for batch_number, data in enumerate(image_loader_validation):
            images, target_labels = data['image'].to(device), data['target_labels'].to(device)
            
            outputs = model(images)
            
            loss = loss_function(outputs, target_labels)
            validation_loss += loss.item()
            
        predictions = output_to_prediction(outputs).cpu()
        true_labels = data["target_labels"].cpu()

        
        all_predictions.extend(predictions.numpy())
        all_true_labels.extend(true_labels.numpy())

        macro_prec, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_true_labels, all_predictions, average='macro', zero_division=1)

        micro_prec, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        all_true_labels, all_predictions, average='micro', zero_division=1)

        avg_validation_loss = validation_loss / (batch_number + 1)

        print(
            f"Validation Loss: {avg_validation_loss:.4f} | "
            f"Macro F1: {macro_f1:.4f} | Macro Precision: {macro_prec:.4f} | Macro Recall: {macro_recall:.4f} | "
            f"Micro F1: {micro_f1:.4f} | Micro Precision: {micro_prec:.4f} | Micro Recall: {micro_recall:.4f}"
)
            
    if early_stopper.early_stop(validation_loss):
        break

# testing
test_predictions = []
test_image_names = []

with torch.no_grad():
    for batch_num, data in enumerate(image_loader_test):
        images, image_names = data['image'].to(device), data['image_name']
        outputs = model(images)
        predictions = output_to_prediction(outputs).cpu()
        
        test_predictions.extend(predictions.numpy())
        test_image_names.extend(image_names)

sorted_indices = sorted(range(len(test_image_names)), key=lambda i: test_image_names[i])
test_image_names = [test_image_names[i] for i in sorted_indices]
test_predictions = [test_predictions[i] for i in sorted_indices]

with open('../data/test_predictions.tsv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')
    writer.writerow(data_loader.header)

    for image_name, prediction in zip(test_image_names, test_predictions):
        row = [os.path.basename(image_name)]
        row_dict = dict(zip(data_loader.header[1:], prediction.astype(int)))
        sorted_prediction = [row_dict[label] for label in data_loader.header[1:]]
        row.extend(sorted_prediction)
        writer.writerow(row)
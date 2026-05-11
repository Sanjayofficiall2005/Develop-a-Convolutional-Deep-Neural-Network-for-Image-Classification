# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
The aim of this experiment is to develop a Convolutional Neural Network (CNN) model to classify images into different categories. The model takes image data as input, learns important features using convolutional layers, and predicts the correct class label. A dataset of labeled images (such as handwritten digits from 0 to 9) is used to train and test the model. Finally, the performance of the model is evaluated, and it is used to predict the class of new unseen images.

## Neural Network Model
<img width="835" height="427" alt="image" src="https://github.com/user-attachments/assets/eaed7ea2-019b-40a8-99b5-34f8b79b91a4" />

## DESIGN STEPS
### STEP 1: 
Load the dataset from the tensorflow library.
### STEP 2: 
Preprocess the dataset.
### STEP 3: 
Create and train your model.
### STEP 4: 
Include the training loss, validation loss vs iteration plot.
### STEP 5: 
Test the model for your handwritten scanned images.
### STEP 6: 
Create and train your model.


## PROGRAM

### Name: SANJAY M

### Register Number: 212223230187

```python

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # write your code here
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,10)


    def forward(self, x):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x
from torchsummary import summary

# Initialize model
model = CNNClassifier()

# Move model to GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)

# Print model summary
print('Name:  SANJAY M ')
print('Register Number: 212223230187 ')
summary(model, input_size=(1, 28, 28))
# Initialize model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
## Step 3: Train the Model
def train_model(model, train_loader, num_epochs=3):
  for epoch in range(num_epochs):
      model.train()
      running_loss = 0.0
      for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # write your code here


      print('Name:SANJAY M  ')
      print('Register Number: 212223230187 ')
      print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')


# Train the model
train_model(model, train_loader)

## Step 4: Test the Model
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print('Name:SANJAY M  ')
    print('Register Number: 212223230187 ')
    print(f'Test Accuracy: {accuracy:.4f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    print('Name:SANJAY M  ')
    print('Register Number: 212223230187 ')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print('Name:SANJAY M  ')
    print('Register Number: 212223230187 ')
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

# Evaluate the model
test_model(model, test_loader)


```

### OUTPUT

## Training Loss per Epoch
<img width="361" height="216" alt="image" src="https://github.com/user-attachments/assets/01f732db-ef12-424c-b1e7-924607db67a5" />


## Confusion Matrix
<img width="729" height="736" alt="image" src="https://github.com/user-attachments/assets/d1d2e831-0144-4c61-be72-355c4d3b504d" />


## Classification Report
<img width="556" height="414" alt="image" src="https://github.com/user-attachments/assets/01e325a2-9018-4b48-a4bd-49e49c515a99" />


### New Sample Data Prediction
<img width="550" height="601" alt="image" src="https://github.com/user-attachments/assets/239ab55f-fb47-45a1-b750-86737fd8b211" />


## RESULT
The Convolutional Neural Network (CNN) model was successfully trained and achieved good classification performance on the given image dataset.

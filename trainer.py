from dataloader import QuickDrawDataSet
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F



def runTrainer():
    
    train_dataset = QuickDrawDataSet("/srv/share/datasets/quickdraw/train")
    test_dataset = QuickDrawDataSet("/srv/share/datasets/quickdraw/test")
    
#     print(test_dataset.__getitem__(20))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=32)

    
    model = models.resnet50(pretrained=False)
    
    model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 57),
                                 nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    model.to(device)
    
    
    epochs = 50
    steps = 0
    running_loss = 0
    print_every = 10
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device,  dtype=torch.float), labels.to(device)
                        logps = model.forward(inputs.to(device, dtype=torch.float))
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()
                    
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                train_losses.append(running_loss/len(trainloader))
                test_losses.append(test_loss/len(testloader)) 
                print(f"Epoch {epoch+1}/{epochs}.. "\
                      f"Train loss: {running_loss/print_every:.3f}.. "\
                      f"Test loss: {test_loss/len(testloader):.3f}.. "\
                      f"Test accuracy: {accuracy/len(testloader):.3f}")
                running_loss = 0
                model.train()
    torch.save(model, 'quickdraw.pth')
    
    
    




if __name__ == "__main__":
    
    runTrainer()
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.utils.data.dataloader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Pad(2)   #add padding to make it 32x32 exactly like in lenet-5
])

train_set = torchvision.datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

validation_set = torchvision.datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)
batch_size=1
training_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)
    
    def forward(self, x):
        x = F.avg_pool2d(F.tanh(self.conv1(x)), (2,2))
        x = F.avg_pool2d(F.tanh(self.conv2(x)), (2,2))
        x = torch.flatten(x, start_dim=1, end_dim=3)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.softmax(self.fc3(x),1)
        return x


model = Network()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

def train_one_epoch():
    running_loss = 0.
    last_loss=0.

    for i, data in enumerate(training_loader):
        input, label = data
        optimizer.zero_grad()
        output = model(input)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        #for bigger batches
        # last_loss = running_loss/(i+1)
        # print('batch {} loss: {}'.format((i+1), last_loss))
        # running_loss = 0

        if i%1000==999:
            last_loss = running_loss/(i+1)
            print('batch {} loss: {}'.format((i+1)/1000, last_loss))
            running_loss = 0
    return last_loss

def train(n):
    epochs = n

    best_vloss = 1000.0

    for epoch in range(epochs):
        print('Epoch {}:'.format(epoch+1))

        model.train(True)
        avg_loss = train_one_epoch()

        running_vloss = 0.0

        model.eval()
        correct = 0
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinput, vlabels = vdata
                voutputs = model(vinput)
                cirterion = nn.CrossEntropyLoss()
                vloss = cirterion(voutputs, vlabels)
                running_vloss +=vloss
                predicted = torch.argmax(voutputs)
                correct += (predicted == vlabels).sum()
        avg_vloss = running_vloss / (i+1)
        correct = (correct / ((i+1)*(batch_size)))*100
        print('Loss train {} valid {} acc %: {}'.format(avg_loss, avg_vloss, correct))
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}'.format(epoch+1)
            torch.save(model.state_dict(), model_path)




def test_trained_model(name):
    saved_model = Network()
    saved_model.load_state_dict(torch.load(name))
    running_vloss = 0
    correct = 0
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
                vinput, vlabels = vdata
                voutputs = saved_model(vinput)
                cirterion = nn.CrossEntropyLoss()
                vloss = cirterion(voutputs, vlabels)
                running_vloss +=vloss
                predicted = torch.argmax(voutputs)
                correct += (predicted == vlabels).sum()
    avg_vloss = running_vloss / (i+1)
    correct = (correct / ((i+1)*(batch_size))) *100
    print('Trained loss: {} Trained acc: {:.2f}%'.format(avg_vloss, correct))

# # training model
# train(1)

test_trained_model('model_10')


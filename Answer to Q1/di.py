import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
import PIL

Batch_size = 50
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Epochs = 10

pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

from torch.utils.data import DataLoader

train_set = datasets.MNIST("data", train=True, download=True, transform=pipeline)

test_set = datasets.MNIST("data", train=False, download=True, transform=pipeline)


train_loader = DataLoader(train_set, batch_size=Batch_size, shuffle=True)

test_loader = DataLoader(test_set, batch_size=Batch_size, shuffle=True)

print("finish")
import os
from skimage import io
import torchvision.datasets.mnist as mnist

root= r"./data/MNIST/raw"
train_set = (
    mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )
test_set = (
    mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
        )
print("training set :",train_set[0].size())
print("test set :",test_set[0].size())
# label: tensor([5, 0, 4,  ..., 5, 6, 8])
train_label = list(train_set[1].numpy())
train_picture = list(train_set[0].numpy())
train_label_lst = []
train_picture_lst = []

for num in range(len(train_label)):
    if train_label[num] == 0 or train_label[num] == 7:
        train_label_lst.append(train_label[num])
        train_picture_lst.append(train_picture[num])

train_images = torch.Tensor(train_picture_lst)
train_labels = torch.Tensor(train_label_lst)


test_label = list(test_set[1].numpy())
test_picture = list(test_set[0].numpy())
test_label_lst = []
test_picture_lst = []

for num in range(len(test_label)):
    if test_label[num] == 0 or test_label[num] == 7:
        test_label_lst.append(test_label[num])

        test_picture_lst.append(test_picture[num])

test_images = torch.Tensor(test_picture_lst)
test_labels = torch.Tensor(test_label_lst)

train_set = (
    train_images,
    train_labels
        )
test_set = (
    test_images,
    test_labels
        )

def convert_to_img(train=True):
    if(train):
        f=open(root+'train.txt','w')
        data_path=root+'/train/'
        if(not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img,label) in enumerate(zip(train_set[0],train_set[1])):
            img_path=data_path+str(i)+'.jpg'
            io.imsave(img_path,img.numpy())
            f.write(img_path+' '+str(label)+'\n')
        f.close()
    else:
        f = open(root + 'test.txt', 'w')
        data_path = root + '/test/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img,label) in enumerate(zip(test_set[0],test_set[1])):
            img_path = data_path+ str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            f.write(img_path + ' ' + str(label) + '\n')
        f.close()

convert_to_img(True) # convert_to_train
convert_to_img(False) # convert_to_test

class MY_MNIST(datasets):
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, transform=None):
        self.transform = transform
        self.data, self.targets = torch.load(root)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)
        img = transforms.ToTensor()(img)

        sample = {'img': img, 'target': target}
        return sample

    def __len__(self):
        return len(self.data)


train = MY_MNIST(root='./mnist/MNIST/processed/training.pt', transform=None)
print(train)
exit()
class Digit(nn.module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20*10*10, 500)
        self.fc2 = nn.Linear(500, 2)

    def forward(self, x):
        input_size = x.size(0)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2,)

        x = self.conv2(x)
        x = F.relu(x)

        x = x.view(input_size, -1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)

        return output


    model = Digit().to(Device)

    optimizer = optim.Adam(model.parameters())

    def train_model(model, device, train_loader, optimizer, epoch):

        model.train()
        for batch_index, (data, target) in enumerate(train_loader):
            data, target = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            pred = output.max(1, keepdim=True)
            loss.backward()
            optimizer.step()
            if batch_index % 3000 == 0:
                print("Train Epoch : {} \t Loss : {:.6f}".format(epoch, loss.item()))

    def test_model(model, device, test_loader):
        model.eval()
        correct = 0.0
        test_loss = 0.0
        num_0 = 0
        num_7 = 0
        with torch.no.grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                output = model(data)
                test_loss += F.cross_entropy(output,target).item()
                preds = output.max(1, keepdim=True)[1]
                for pred in preds.numpy():
                    if pred == 0:
                        num_0 += 1
                    if pred == 7:
                        num_7 += 1
            proportion_0 = num_0 / data.size()[0]
            proportion_7 = num_7 / data.size()[0]
        return proportion_0, proportion_7




import numpy as np
import pickle as pkl
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




#########PARAMETERS

dtype = torch.float
device = torch.device("cpu")
print_interval = 10
img_width = 28
total_data_size = 60000
train_data_amount = 50000
test_data_amount = 10000


#########
# HYPERPARAMETERS

learning_rate = 0.1
momentum = 0.5 # for optimizer
epochs = 3
batch_size = 64


class Model(nn.Module):



    def __init__(self):

        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)


    def forward(self, x):

        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = x.view(-1, 320)

        x = F.relu(self.fc1(x))

        x = F.dropout(x, training=self.training)

        x = self.fc2(x)

        return F.log_softmax(x)



model = Model()

optimizer = optim.SGD(model.parameters(), lr=learning_rate,

                      momentum=momentum)






class Trainer:



    def __init__(self, dataset):

        self.dataset = dataset



    def train(self, epoch):

      model.train()

      idx = 0

      for data, target in self.dataset.trainData:

        optimizer.zero_grad()

        output = model(data)

        loss = F.nll_loss(output, target)

        loss.backward()

        optimizer.step()



        if idx % print_interval == 0:

          print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

            epoch, idx * len(data), train_data_amount,

            100. * idx / len(self.dataset.trainData), loss.item()))

        idx += 1



    def test(self):

      model.eval()

      test_loss = 0

      correct = 0

      with torch.no_grad():

        for data, target in self.dataset.testData:

            output = model(data)

            test_loss += F.nll_loss(output, target, reduction='sum').item()

            pred = output.data.max(1, keepdim=True)[1]

            correct += pred.eq(target.data.view_as(pred)).sum()
            

      test_loss /= test_data_amount

      print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(

        test_loss, correct, test_data_amount,

        100. * correct / test_data_amount))





class Digits_Data:



    def __init__(self, relative_path, data_file_name='digits_data.pkl'):

        self.index = -1

        with open(relative_path + data_file_name, "rb") as d:

            data = self.format_data(pkl.load(d)['train'])
            

            self.trainData = self.batchData(data[:train_data_amount])

            self.testData = self.batchData(data[-test_data_amount:])



    def format_data(self, rawData):

        formatted_data = []

        # Change format of rawData

        for num in rawData:

            for img in rawData[num]:

                img = img / np.linalg.norm(img) 

                formatted_data.append((img, num))

        random.shuffle(formatted_data)

        return formatted_data



    def batchData(self, formatted_data):

        data = []

        imgs, targets = zip(*formatted_data)

        i = 0

        while i < len(formatted_data):

            # convert to tensor and put in batch

            batch = imgs[i : i + batch_size]

            target = targets[i : i + batch_size]

            batch = torch.tensor(batch, dtype=torch.float)

            target = torch.tensor(target, dtype=torch.long)

            if (len(batch) < batch_size): # only use batches of full size

                break

            batch = batch.view(batch_size, 1, img_width, img_width)

            data.append((batch, target))

            i += batch_size

        return data



def main():



    data = Digits_Data(relative_path='/Users/guillaumedelande/Documents/AIGroupWork/stephenfitz.keio2019aia/keio2019aia/data/assignment1/')
    trainer = Trainer(data)

    trainer.test()

    for epoch in range(1, epochs + 1):

      trainer.train(epoch)

      trainer.test()



if __name__ == '__main__':

    main()


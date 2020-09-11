import torch
import torch.nn as nn
from dataset import make_train_data_loader
from torchvision import transforms
from model import VGG11
import copy
import matplotlib.pyplot as plt
import numpy as np

train_data_path = '/home/ykwei/Documents/Kaggle-Plant-Seedlings-Classification/train'
weight_path = '/home/ykwei/Documents/svm_loss/weights/model.pth'
use_cuda = True
gpu_id = 5
num_epochs = 150
valid_size = 0.2

def train():
    model = VGG11()
    if use_cuda:
        torch.cuda.set_device(gpu_id)
        model = model.cuda()

    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_loader, valid_loader = make_train_data_loader(train_data_path, data_transform)
    print('trainset len:', len(train_loader.dataset))
    print('train loader len:', len(train_loader))
    print('valid loader len:', len(valid_loader))
    print('=========================================')
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)
    SVMloss = nn.MultiMarginLoss()

    train_loss_list = []
    valid_loss_list = []
    train_accuracy_list =[]
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1}/{num_epochs}')
        print('-' * len(f'Epoch: {epoch + 1}/{num_epochs}'))

        train_loss = 0.0
        valid_loss = 0.0
        training_accuracy = 0.0
        predict_correct = 0
        for data, label in train_loader:
            if use_cuda:
                data, label = data.cuda(), label.cuda()
            optimizer.zero_grad()
            output = model(data)
            _, prediction = torch.max(output.data, 1)
            loss = SVMloss(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            predict_correct += torch.sum(prediction == label.data)

        model.eval()
        for data, label in valid_loader:
            if use_cuda:
                data, label = data.cuda(), label.cuda()
            output = model(data)
            loss = SVMloss(output, label)
            valid_loss += loss.item() * data.size(0)

        train_loss = train_loss / float(np.floor(len(train_loader.dataset) * (1 - valid_size)))
        train_loss_list.append(train_loss)
        valid_loss = valid_loss / float(np.floor(len(valid_loader.dataset) * valid_size))
        valid_loss_list.append(valid_loss)

        training_accuracy = float(predict_correct) / float(len(train_loader.dataset))
        train_accuracy_list.append(training_accuracy)
        print(f'Training loss: {train_loss:.4f}\nValidation loss: {valid_loss:.4f}\nAccuracy: {training_accuracy:.4f}')

        if training_accuracy > best_accuracy:
            best_accuracy = training_accuracy
            # torch.save(model.state_dict(), weight_path)
            best_weight = copy.deepcopy(model.state_dict())
            print(f'Best accuracy update, current best weights saved')
            # print(f'best accuracy update: {best_accuracy:.4f}, current best weights saved')
        print('\n')

    # model.load_state_dict(best_weight)
    torch.save(best_weight, weight_path)
    print(f'best weight saved at {weight_path}')

    x1 = range(0,len(train_accuracy_list))
    x2 = range(0,len(train_loss_list))
    x3 = range(0,len(valid_loss_list))
    y1 = train_accuracy_list
    y2 = train_loss_list
    y3 = valid_loss_list
    plt.subplots_adjust(left = 0.1, bottom = 0.2, right = 0.9, top = 0.9, wspace = 0.1, hspace = 0.9)
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'm', linestyle='-', label='Training accuracy')
    plt.xlabel(u'epoches')
    plt.ylabel(u'Accuracy')
    plt.xlim(0,len(train_accuracy_list))
    plt.title('Train accuracy vs. epoches')
    plt.grid('on')

    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, 'c', linestyle='-', label='train loss')
    plt.plot(x3, y3, 'y', linestyle='-', label='valid loss')
    plt.xlabel(u'epoches')
    plt.ylabel(u'Loss')
    plt.xlim(0,len(train_accuracy_list))
    plt.title('Train loss vs. valid loss')
    plt.grid('on')
    plt.legend(loc=1)

    plt.savefig("accuracy_loss.png")
    plt.show()

if __name__ == '__main__':
    train()
    print('train end')

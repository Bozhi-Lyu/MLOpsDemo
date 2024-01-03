import torch
from torch.utils.data import TensorDataset

def corruptmnist():
    
    filepath = './corruptmnist/'
    all_images = []
    all_targets = []

    for i in range(6):
        images = torch.load(filepath + 'train_images_{}.pt'.format(i))
        targets = torch.load(filepath + 'train_target_{}.pt'.format(i))
        all_images.append(images)
        all_targets.append(targets)

    train_images = torch.cat(all_images, dim = 0).unsqueeze(1)
    train_targets = torch.cat(all_targets, dim = 0)
    # one_hot_train_targets = F.one_hot(train_targets.squeeze(), num_classes=10).float()
    test_images = torch.load(filepath + 'test_images.pt').unsqueeze(1)
    test_targets = torch.load(filepath + 'test_target.pt')
    # one_hot_test_targets = F.one_hot(test_targets.squeeze(), num_classes=10).float()

    train_dataset = TensorDataset(train_images, train_targets)
    test_dataset = TensorDataset(test_images, test_targets)
    # batch_size = 20
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, shuffle=False)

    return train_dataset, test_dataset

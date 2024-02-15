import os
import copy
from config import device
import config
from collections import defaultdict
import torch
import torch.utils.data
from helper import Helper
import random
import logging
from torchvision import datasets, transforms
import numpy as np

logger = logging.getLogger("logger")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class PartDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, lab_list):
        self.dataset = dataset
        self.used_ids = []
        self.lab_list = lab_list
        self.lab_map = dict()
        for i in range(len(lab_list)):
            self.lab_map[lab_list[i]] = i
        print("lab_map", self.lab_map)
        for i, (X, y) in enumerate(dataset):
            if y in self.lab_list:
                self.used_ids.append(i)

    def __len__(self,):
        return len(self.used_ids)

    def __getitem__(self, i):
        X, y = self.dataset[self.used_ids[i]]
        y_new = self.lab_map[y]
        return X, y_new


def get_resnet18(pretrained=True, num_of_classes=10):

    from torchvision import models
    import torch.nn as nn
    model_ft = models.resnet18(pretrained=pretrained)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_of_classes)
    return model_ft


def get_cifarcnn(num_of_classes=10):

    import torch.nn as nn
    from models.CifarNet import CifarNet100

    model = CifarNet100().cuda()
    loaded_params = torch.load('./saved_models/cifar100_model_best.pt.tar')
    model.load_state_dict(loaded_params['state_dict'])
    model.fc = nn.Linear(128, num_of_classes, bias=True).cuda()

    return model


class ImageHelper(Helper):

    def create_one_model(self):
        local_model = None
        if self.params['type'] == config.TYPE_CIFAR:
            if self.params['binary_cls'] == True:
                num_of_classes = 2
            else:
                num_of_classes = 10
            local_model = get_cifarcnn(num_of_classes)

        elif self.params['type'] == config.TYPE_MNIST:
            from models.MnistNet import MnistNet
            if self.params['binary_cls'] == True:
                num_of_classes = 2
            else:
                num_of_classes = 10

            local_model = MnistNet(name='Local',
                                   created_time=self.params['current_time'], num_of_classes=num_of_classes)

        local_model = local_model.to(device)
        return local_model

    def create_model(self):
        local_model = None
        target_model = None
        if self.params['type'] == config.TYPE_CIFAR:
            if self.params['binary_cls'] == True:
                num_of_classes = 2
            else:
                num_of_classes = 10

            local_model = get_cifarcnn(num_of_classes)
            target_model = get_cifarcnn(num_of_classes)

        elif self.params['type'] == config.TYPE_MNIST:
            from models.MnistNet import MnistNet
            if self.params['binary_cls'] == True:
                num_of_classes = 2
            else:
                num_of_classes = 10
            local_model = MnistNet(name='Local',
                                   created_time=self.params['current_time'], num_of_classes=num_of_classes)
            target_model = MnistNet(name='Target',
                                    created_time=self.params['current_time'], num_of_classes=num_of_classes)

        local_model = local_model.to(device)
        target_model = target_model.to(device)
        if self.params['resumed_model']:
            if torch.cuda.is_available():
                loaded_params = torch.load(
                    f"{self.params['resumed_model_name']}")
            else:
                loaded_params = torch.load(
                    f"{self.params['resumed_model_name']}", map_location='cpu')
            target_model.load_state_dict(loaded_params['state_dict'])
            self.start_epoch = loaded_params['epoch']+1
            self.params['lr'] = loaded_params.get('lr', self.params['lr'])
            logger.info(f"Loaded parameters from saved model: LR is"
                        f" {self.params['lr']} and current epoch is {self.start_epoch}")
        else:
            self.start_epoch = 1

        self.local_model = local_model
        self.target_model = target_model

    def build_classes_dict(self):
        classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if label in classes:
                classes[label].append(ind)
            else:
                classes[label] = [ind]
        return classes

    def sample_dirichlet_train_data(self, no_participants, alpha=0.9, lst_sample=2):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.
        """

        classes = self.classes_dict
        per_participant_list = defaultdict(list)
        num_classes = len(classes.keys())  # for cifar: 10
        image_nums = []
        for n in range(num_classes):
            image_num = []
            random.shuffle(classes[n])
            class_size = len(classes[n]) - no_participants*lst_sample
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))+lst_sample
                sampled_list = classes[n][:min(len(classes[n]), no_imgs)]
                image_num.append(len(sampled_list))
                per_participant_list[user].extend(sampled_list)
                classes[n] = classes[n][min(len(classes[n]), no_imgs):]
            image_nums.append(image_num)
            print(n, image_nums[n])

        return per_participant_list

    def poison_test_dataset(self):
        logger.info('get poison test loader')
        # delete the test data with target label
        test_classes = {}
        for ind, x in enumerate(self.test_dataset):
            _, label = x
            if label in test_classes:
                test_classes[label].append(ind)
            else:
                test_classes[label] = [ind]

        range_no_id = list(range(0, len(self.test_dataset)))
        for image_ind in test_classes[self.params['poison_label_swap']]:
            if image_ind in range_no_id:
                range_no_id.remove(image_ind)
        poison_label_inds = test_classes[self.params['poison_label_swap']]

        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               range_no_id)), \
            torch.utils.data.DataLoader(self.test_dataset,
                                        batch_size=self.params['batch_size'],
                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                            poison_label_inds))

    def load_data(self):
        logger.info('Loading data')
        dataPath = "./data"
        if self.params['type'] == config.TYPE_CIFAR:
            # data load
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            self.train_dataset = datasets.CIFAR10(dataPath, train=True, download=True,
                                                  transform=transform_train)

            self.test_dataset = datasets.CIFAR10(
                dataPath, train=False, transform=transform_test)
            if self.params['binary_cls'] == True:
                self.LABEL_LIST = [0, 2]
                self.train_dataset = PartDataset(
                    self.train_dataset, self.LABEL_LIST)
                self.test_dataset = PartDataset(
                    self.test_dataset, self.LABEL_LIST)

            print("cifar load data done")

        elif self.params['type'] == config.TYPE_MNIST:

            self.train_dataset = datasets.MNIST(dataPath, train=True, download=True,
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                ]))
            self.test_dataset = datasets.MNIST(dataPath, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
            ]))

            if self.params['binary_cls'] == True:
                self.LABEL_LIST = [0, 1]
                self.train_dataset = PartDataset(
                    self.train_dataset, self.LABEL_LIST)
                self.test_dataset = PartDataset(
                    self.test_dataset, self.LABEL_LIST)

        self.classes_dict = self.build_classes_dict()

        if self.params['sampling_dirichlet']:
            logger.info('Dirichlet')
            # sample indices for participants using Dirichlet distribution
            indices_per_participant = self.sample_dirichlet_train_data(
                self.params['number_of_total_participants'],
                alpha=self.params['dirichlet_alpha'])
            train_loaders = [(pos, self.get_train(indices)) for pos, indices in
                             indices_per_participant.items()]
        else:
            # sample indices for participants that are equally
            logger.info('iid')
            all_range = list(range(len(self.train_dataset)))
            random.shuffle(all_range)
            train_loaders = [(pos, self.get_train_iid(all_range, pos))
                             for pos in range(self.params['number_of_total_participants'])]

        self.train_data = train_loaders
        self.test_data = self.get_test()

        if self.params['is_poison'] == True:
            self.test_data_poison, self.test_targetlabel_data = self.poison_test_dataset()

    def get_train_iid(self, all_range, model_no):
        """
        This method equally splits the dataset.
        :param params:
        :param all_range:
        :param model_no:
        :return:
        """

        data_len = int(len(self.train_dataset) /
                       self.params['number_of_total_participants'])
        sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
        if self.params['batch_size'] == -1:
            logger.info('new batch size='+str(len(sub_indices)))
            self.params['batch_size'] = len(sub_indices)
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.params['batch_size'],
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                       sub_indices))
        return train_loader

    def get_test(self):
        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=self.params['test_batch_size'],
                                                  shuffle=False)
        return test_loader

    def get_batch(self, train_data, bptt, evaluation=False):
        data, target = bptt
        data = data.to(device)
        target = target.to(device)
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        return data, target

    def get_poison_batch(self, bptt, adversarial_index=0, evaluation=False, agent_name=-1):
        # adversarial_index =1 backdoor
        # adversarial_index =0 label-flipping
        images, targets = bptt

        poison_count = 0
        new_images = copy.deepcopy(images)
        new_targets = copy.deepcopy(targets)

        for index in range(0, len(images)):
            if evaluation:  # poison all data when testing
                new_targets[index] = self.params['poison_label_swap']
                if adversarial_index == 1:
                    new_images[index] = self.add_pixel_pattern(images[index])

                poison_count += 1

            else:  # poison part of data when training
                if index < self.params['poisoning_per_batch']:
                    new_targets[index] = self.params['poison_label_swap']
                    if adversarial_index == 1:
                        new_images[index] = self.add_pixel_pattern(
                            images[index], agent_name)
                    poison_count += 1

        new_images = new_images.to(device)
        new_targets = new_targets.to(device).long()
        if evaluation:
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)
        return new_images, new_targets, poison_count

    def add_pixel_pattern(self, ori_image, agent_name=-1):
        image = ori_image
        poison_patterns = self.params['poison_pattern']
        if agent_name != -1:
            poison_patterns = self.params[str(agent_name)+'_poison_pattern']

        if self.params['type'] == config.TYPE_CIFAR:
            for i in range(0, len(poison_patterns)):
                pos = poison_patterns[i]
                image[0][pos[0]][pos[1]] = 1
                image[1][pos[0]][pos[1]] = 1
                image[2][pos[0]][pos[1]] = 1

        elif self.params['type'] == config.TYPE_MNIST:
            for i in range(0, len(poison_patterns)):
                pos = poison_patterns[i]

                image[0][pos[0]][pos[1]] = 1

        return image

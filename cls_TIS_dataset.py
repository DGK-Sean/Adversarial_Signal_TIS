import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class TISDataset(Dataset):
    def read_dna_file(self, file_location, class_id):
        current_file = open(file_location)
        dna_list = []

        for line in current_file:
            stripped_line = line.strip()
            # Ignoring all the '' in the text file
            if stripped_line == '':
                pass
            else:
                dna_list.append((stripped_line, class_id))

        return dna_list

    def __init__(self, pos_data_loc, neg_data_loc, transform=None):
        # Initialization
        self.dna_list = self.read_dna_file(pos_data_loc, 1)
        self.dna_list.extend(self.read_dna_file(neg_data_loc, 0))
        self.data_len = len(self.dna_list)
        self.transform = transform

    def __getitem__(self, index):
        # Read data
        dna_data, label = self.dna_list[index]
        # Convert AGCT to 0123
        dna_data = dna_data.replace('A', '0')
        dna_data = dna_data.replace('G', '1')
        dna_data = dna_data.replace('C', '2')
        dna_data = dna_data.replace('T', '3')
        dna_data = dna_data.replace('N', '0')
        dna_data = dna_data.replace('R', '1')
        dna_data = dna_data.replace('Y', '3')
        dna_data = dna_data.replace('K', '3')
        dna_data = dna_data.replace('S', '2')

        '''
        from Yunseol
        '''
        # quick 'n dirty: treating N as A - they are very uncommon anyway
        # treating Y as T - they are very uncommon and more frequent T than C in NF
        # treating K as T - they are very uncommon and more frequent T than G in NF
        # treating R as A - they are very uncommon and more frequent A than G in NF
        # treating S as G - they are very uncommon and more frequent G than C in NF?
        # Convert string numbers to int and convert it to ndarray
        dna_data = np.asarray([int(digit) for digit in dna_data])
        # print(dna_data)
        # Create one hot encoding, row x column
        one_hot_dna_data = np.zeros((len(dna_data), 4), dtype=int)
        # print(one_hot_dna_data)

        # Assign values
        rows = np.arange(dna_data.size)
        # Convert the nucleotides to one hot encoding
        # A = [1 0 0 0]
        # G = [0 1 0 0]
        # C = [0 0 1 0]
        # T = [0 0 0 1]
        one_hot_dna_data[rows, dna_data] = 1
        #print(one_hot_dna_data)

        # Convert numpy to tensor
        dna_data_as_ten = torch.from_numpy(one_hot_dna_data).float()
        # 여기서 unsqueeze 한번더
        dna_data_as_ten = dna_data_as_ten.unsqueeze(0)
        return dna_data_as_ten, label, index
        # Return index too

    def __len__(self):
        return self.data_len


if __name__ == '__main__':
    '''
    BATCH_SIZE = 64
    transformations = transforms.Compose([transforms.ToTensor()])

    train_TIS_dataset = TISDataset('C:/Users/dongg/Desktop/bsc_project/Thesis_Project_Files/dataset'
                                   '/raw_data/human/train/CCDS60-140.pos',
                                   'C:/Users/dongg/Desktop/bsc_project/Thesis_Project_Files/'
                                   'dataset/edited_data/human_balanced/CCDS60-140_balanced.neg')

    test_TIS_dataset = TISDataset('C:/Users/dongg/Desktop/bsc_project/Thesis_Project_Files/dataset/'
                                  'raw_data/human/test/chrom21_60-140.pos',
                                  'C:/Users/dongg/Desktop/bsc_project/Thesis_Project_Files/dataset/'
                                  'edited_data/human_balanced/chrom21_60-140_balanced.neg', transformations)

    train_loader = torch.utils.data.DataLoader(dataset=train_TIS_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_TIS_dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True)


    # Define transforms
    for dna, label in dna_dataset_loader:
        # Feed the data to the model
        pass
    train_set = TISDataset(pos_data_loc, neg_data_loc)
    test_set = TISDataset(pos_data_loc, neg_data_loc)

    my_dataset =     TISDataset(pos_data_loc, neg_data_loc)
    print(my_dataset, '1')
    data_index = 13
    single_dna_data, label = my_dataset.__getitem__(data_index)
    print(single_dna_data.size())
    single_dna_data.unsqueeze_(0)
    single_dna_data.unsqueeze_(0)
    print(single_dna_data.size())
    pos_data_loc = 'CCDS60-140.pos'
    neg_data_loc = 'CCDS60-140.neg'
    my_dataset = TISDataset('short_pos.pos', 'short_neg.neg')
    print(my_dataset, '1')
    data_index = 13
    single_dna_data, label = my_dataset.__getitem__(data_index)
'''
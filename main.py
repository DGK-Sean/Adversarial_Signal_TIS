import torch
from cls_TIS_model_sequential import Net
from cls_TIS_dataset import TISDataset
from fcn_train_model import train, evaluate
from fcn_checkpoint import save_checkpoint, save_last, save_best_model
from fcn_graph import graph_maker
from fcn_csv import csv_writer, max_acc_csv
from torchvision import transforms
from cls_TIS_model_utku import TISv1
from cls_TIS_model_exp3 import Net3


if __name__ == "__main__":
    epochs = 130
    BATCH_SIZE = 1024

    # CUDA for PyTorch
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

    # Transformation
    transformations = transforms.Compose([transforms.ToTensor()])

    # Dataset
    # fileInLongPath = os.path.join(os.getcwd(), os.listdir(os.getcwd())[0])
    # this will get the first file in the last directory of your path
    # os.path.dirname(fileInLongPath) # this will get directory of file
    # os.path.dirname(os.path.dirname(fileInLongPath)) # this will get the directory of the directory of the file

    train_TIS_dataset = TISDataset('C:/Users/dongg/Desktop/bsc_project/Thesis_Project_Files/dataset'
                                   '/raw_data/human/train/CCDS60-140.pos',
                                   'C:/Users/dongg/Desktop/bsc_project/Thesis_Project_Files/'
                                   'dataset/edited_data/human_balanced/CCDS60-140_balanced.neg')

    test_TIS_dataset = TISDataset('C:/Users/dongg/Desktop/bsc_project/Thesis_Project_Files/dataset/'
                                  'raw_data/human/test/chrom21_60-140.pos',
                                  'C:/Users/dongg/Desktop/bsc_project/Thesis_Project_Files/dataset/'
                                  'edited_data/human_balanced/chrom21_60-140_balanced.neg', transformations)

    # Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_TIS_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_TIS_dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False)

    # Model and Training

    model = Net().to(DEVICE)
    #model = Net3().to(DEVICE)
    #model = TISv1().to(DEVICE)

    # Variables Declaration
    l_rate = 0.05
    moment = 0.9
    nest = 'TRUE'
    model_number = 28
    wd = 0  # Weight decay
    best_acc = 90  # Preventing the first accuracy to be saved. Normally it is a low value
    optim_csv = 'SGD'

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=l_rate, momentum=moment, nesterov=True, weight_decay=wd)
    #optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)

    # loss function
    criterion = torch.nn.CrossEntropyLoss()

    # List to Keep Track
    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []

    # Labels Separation Accuracy Check Lists
    train_label_0_acc_list = []
    train_label_1_acc_list = []
    test_label_0_acc_list = []
    test_label_1_acc_list = []

    epoch_list = []
    csv_write_list = []
    max_list = []

    # Variables for csv writer
    '''
    Always change learning rate/momentum/nesterov/model_number for different models 
    Some Notes: Added Weight Decay to the parameters
    '''

    # print statistics for train
    for epoch in range(1, epochs+1):
        train(model, train_loader, optimizer, criterion, epoch)
        train_loss, train_acc, train_label_0_acc, train_label_1_acc = evaluate(model, train_loader)

        print('[{}] Train Loss: {:.4f}, Train Accuracy: {:.2f}%'.format(
            epoch, train_loss, train_acc))

        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        train_label_0_acc_list.append(train_label_0_acc)
        train_label_1_acc_list.append(train_label_1_acc)
        epoch_list.append(epoch)

        print('test number:', epoch)

        # Evaluate the trained model using the test set/loader
        test_loss, test_acc, test_label_0_acc, test_label_1_acc = evaluate(model, test_loader)
        print('[{}] Test Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(
            epoch, test_loss, test_acc))

        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)
        test_label_0_acc_list.append(test_label_0_acc)
        test_label_1_acc_list.append(test_label_1_acc)

        # Try to save the best model with best test accuracy
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_checkpoint = {'epoch': epoch, 'state_dict': model.state_dict(),
                                     'best_acc': best_acc,
                                     'optimizer': optimizer.state_dict()}
            save_best_model(best_model_checkpoint)

        # save in the existing csv by appending new row
        csv_write_list += (model_number, epoch, l_rate, moment, nest, wd, BATCH_SIZE, train_acc, train_label_0_acc,
                           train_label_1_acc, test_acc, test_label_0_acc, test_label_1_acc, optim_csv,
                           train_loss, test_loss, 'following model8')
        csv_writer(csv_write_list)

        # Make an empty list to reset the appending process of the next epoch/train/test and other parameters
        csv_write_list = []

        # Save the model for every epoch
        if epoch == 1:
            checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_checkpoint(checkpoint)

    '''
    Here I want to find the max value of the test accuracy and their components in the index 
    If test accuracy is more important than train accuracy, my approach is correct (?)
    '''
    max_test_acc = max(test_acc_list)
    max_index = test_acc_list.index(max_test_acc)

    # Another csv file where it only contains max test acc value and their corresponding parameters
    # CSV file name is max_model_tracker_edited.csv
    max_list += (model_number, epoch_list[max_index], l_rate, moment, nest, wd, BATCH_SIZE,
                 train_acc_list[max_index], train_label_0_acc_list[max_index], train_label_1_acc_list[max_index],
                 max_test_acc, test_label_0_acc_list[max_index], test_label_1_acc_list[max_index], optim_csv,
                 train_loss_list[max_index], test_loss_list[max_index], 'following model8')
    max_acc_csv(max_list)
    print(max_test_acc, 'Max_test_accuracy')
    print(max_index, 'max index of test')
    print(test_acc_list[max_index], 'using index method')

    # Make and Save Graph
    graph_maker(train_acc_list, test_acc_list, epoch_list, model_number)

    # Saving it as GPU or CPU does not matter. When you load with cpu code, it loads as cpu code
    # That is what I found in Google :o !

    last_model = model.state_dict()
    save_last(last_model)

    '''    
    # Load on GPU 
    device = torch.device("cuda")
    model.load_state_dict(torch.load(PATH))
    model.to(device)
    
    # Load on CPU
    device = torch.device('cpu')
    model = Net()
    model.load_state_dict(torch.load(PATH, map_location=device))
    '''

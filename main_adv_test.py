import torch
from cls_TIS_model_sequential import Net
from cls_TIS_dataset import TISDataset
from torchvision import transforms
from fcn_get_pred import get_collapsed_pred, average_pred, st_dev_pred, median_pred, graph_mean_median_preds, each_nucleotide
from fcn_adv_accuracy import adversarial_acc_pred
if __name__ == "__main__":
    epochs = 50
    BATCH_SIZE = 64

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

    test_loader = torch.utils.data.DataLoader(dataset=test_TIS_dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False)

    model = Net()
    device = torch.device('cpu')

    #get_collapsed_pred(test_TIS_dataset, 'final_model20.pth', 1)
    each_nucleotide(test_TIS_dataset, 'final_model20.pth', 0)
    #each_nucleotide(test_TIS_dataset, 'final_model20.pth', 1)
    #get_collapsed_pred(test_TIS_dataset, 'final_model20.pth', 0)
    #average_pred(test_TIS_dataset, 'final_model20.pth', 0)
    #st_dev_pred(test_TIS_dataset, 'final_model20.pth', 0)
    #median_pred(test_TIS_dataset, 'final_model20.pth', 1)
    #graph_mean_median_preds(test_TIS_dataset, 'final_model20.pth', 0)
    #adversarial_acc_pred(test_loader, 'final_model20.pth')




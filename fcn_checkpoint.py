import torch

'''
This gives a checkpoint at a certain epoch
'''


def save_checkpoint(state, filename="my_checkpoint_model28.pth"):
    print("=> saving checkpoint for every epoch")
    torch.save(state, filename)


def save_best_model(state, filename="best_acc_model28.pth"):
    print("~~~Saving Best model with accuracy~~~")
    torch.save(state, filename)


def save_last(state, filename="final_model28.pth"):
    print("saving last model")
    torch.save(state, filename)
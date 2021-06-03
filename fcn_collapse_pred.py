import torch
from cls_TIS_model_sequential import Net
from cls_TIS_dataset import TISDataset


model = Net()
device = torch.device('cpu')
from fcn_graph import adv_graph


def get_collapsed_pred(one_data, one_model, target_class):
    model.load_state_dict(torch.load(one_model, map_location=device))
    model.eval()
    print(len(one_data))
    for data_point in range(len(one_data)):
        print(data_point, 'data')
        x, y, z = one_data[data_point]
        print(x)
        print(y)
        print(z)
        '''
        if y == target_class:
            x = x.unsqueeze(0)
            print(x)
            x.requires_grad = True
            out = model(x)
            one_hot_output = torch.FloatTensor(1, out.size()[-1]).zero_()
            one_hot_output[0][1] = 1
            # Backward pass
            out.backward(gradient=one_hot_output)
            # Convert Pytorch variable to numpy array
            # [0] to get rid of the first channel (1,3,224,224)
            adv_signal = x.grad * 50
            x = x + adv_signal

            out = model(x)
            one_dimension = torch.sum(abs(x), axis=(0, 1, 3))
            nparray = one_dimension.detach().numpy()
            #print(nparray)
            print(y, 'target')
            #print(target_class)
        #adv_graph(nparray)
        '''
        return data_point




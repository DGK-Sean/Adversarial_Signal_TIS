import torch
import numpy as np
import statistics
from cls_TIS_model_sequential import Net
from fcn_graph import adv_graph_one, plot_two_graph
from fcn_find_max import find_max, make_absolute
from fcn_remove_dtype import remove_dtype
model = Net()
device = torch.device('cpu')


def get_collapsed_pred(data, one_model, target_class):
    model.load_state_dict(torch.load(one_model, map_location=device))

    model.eval()
    datapoint_list = []
    for point in data:
        x, y, z = point
        if y == target_class:
            x = x.unsqueeze(0)
            x.requires_grad = True
            out = model(x)
            one_hot_output = torch.FloatTensor(1, out.size()[-1]).zero_()
            one_hot_output[0][1] = 1
            # Backward pass
            out.backward(gradient=one_hot_output)
            # Convert Pytorch variable to numpy array
            # [0] to get rid of the first channel (1,3,224,224)
            adv_signal = x.grad
            print(adv_signal,'adv signal')
            x = x + adv_signal
            print(x)
            out = model(x)
            one_dimension = torch.sum(abs(x), axis=(0, 1, 3))
            nparray = one_dimension.detach().numpy()

            # FIX ME !!!!!!!!!!!!!!!!!!!!!!!!!!!
            #datapoint_list.append(nparray/np.max(nparray))
            datapoint_list.append(nparray)

    print(np.asarray(datapoint_list), 'nparray')
    print(np.max(datapoint_list), 'npmax')
    normalized_datapoint_list = np.asarray(datapoint_list) / np.max(datapoint_list)
    print(normalized_datapoint_list, 'hello')
    # adv_graph(normalized_datapoint_list)
    #normalized_datapoint_list = datapoint_list
    return normalized_datapoint_list


def each_nucleotide(data, one_model, target_class):
    model.load_state_dict(torch.load(one_model, map_location=device))
    model.eval()
    # A = [1 0 0 0]
    # G = [0 1 0 0]
    # C = [0 0 1 0]
    # T = [0 0 0 1]

    # Initialize parameters of pos neg list
    a_pos_list = [[] for _ in range(203)]
    a_neg_list = [[] for _ in range(203)]
    g_pos_list = [[] for _ in range(203)]
    g_neg_list = [[] for _ in range(203)]
    c_pos_list = [[] for _ in range(203)]
    c_neg_list = [[] for _ in range(203)]
    t_pos_list = [[] for _ in range(203)]
    t_neg_list = [[] for _ in range(203)]

    for point in data:
        x, y, z = point
        if y == target_class:
            x = x.unsqueeze(0)
            x.requires_grad = True
            out = model(x)
            one_hot_output = torch.FloatTensor(1, out.size()[-1]).zero_()
            one_hot_output[0][1] = 1
            # Backward pass
            out.backward(gradient=one_hot_output)
            # Convert Pytorch variable to numpy array
            # [0] to get rid of the first channel (1,3,224,224)
            adv_signal = x.grad
            x = x + adv_signal
            out = model(x)
            reshape_dimension = x.detach().numpy()
            reshape_dimension = np.reshape(reshape_dimension, (-1, 4))

            # Making the list for A T G C pos/neg separately
            # Comment Here: do we need to include 0 in the pos_list as well? @utku
            for position, rows in enumerate(reshape_dimension):
                if rows[0] >= 0:
                    a_pos_list[position].append(rows[0])
                if rows[0] < 0:
                    a_neg_list[position].append(rows[0])
                if rows[1] >= 0:
                    g_pos_list[position].append(rows[1])
                if rows[1] < 0:
                    g_neg_list[position].append(rows[1])
                if rows[2] >= 0:
                    c_pos_list[position].append(rows[2])
                if rows[2] < 0:
                    c_neg_list[position].append(rows[2])
                if rows[3] >= 0:
                    t_pos_list[position].append(rows[3])
                if rows[3] < 0:
                    t_neg_list[position].append(rows[3])

    retain_a_neg_list = a_neg_list
    retain_t_neg_list = t_neg_list
    retain_g_neg_list = g_neg_list
    retain_c_neg_list = c_neg_list

    # before finding the max values in the negative, I make the values all absolute
    abs_a_neg_list = make_absolute(retain_a_neg_list)
    abs_t_neg_list = make_absolute(retain_t_neg_list)
    abs_g_neg_list = make_absolute(retain_g_neg_list)
    abs_c_neg_list = make_absolute(retain_c_neg_list)

    # Finding the max value of each A T G C pos/neg for normalization
    max_value_A_pos = find_max(a_pos_list)
    max_value_A_neg = find_max(abs_a_neg_list)
    max_value_T_pos = find_max(t_pos_list)
    max_value_T_neg = find_max(abs_t_neg_list)
    max_value_C_pos = find_max(c_pos_list)
    max_value_C_neg = find_max(abs_c_neg_list)
    max_value_G_pos = find_max(g_pos_list)
    max_value_G_neg = find_max(abs_g_neg_list)

    for i in range(len(a_pos_list)):
        a_pos_list[i] = np.asarray(a_pos_list[i])
        a_neg_list[i] = np.asarray(a_neg_list[i])
        t_pos_list[i] = np.asarray(t_pos_list[i])
        t_neg_list[i] = np.asarray(t_neg_list[i])

        c_pos_list[i] = np.asarray(c_pos_list[i])
        c_neg_list[i] = np.asarray(c_neg_list[i])

        g_pos_list[i] = np.asarray(g_pos_list[i])
        g_neg_list[i] = np.asarray(g_neg_list[i])

    # When you normalize, you get still a positive value for all the
    # negative values in the data as you divide with the neg. value
    normalized_a_pos_list = np.asarray(a_pos_list) / max_value_A_pos
    normalized_a_neg_list = np.asarray(a_neg_list) / max_value_A_neg

    normalized_t_pos_list = np.asarray(t_pos_list) / max_value_T_pos
    normalized_t_neg_list = np.asarray(t_neg_list) / max_value_T_neg

    normalized_c_pos_list = np.asarray(c_pos_list) / max_value_C_pos
    normalized_c_neg_list = np.asarray(c_neg_list) / max_value_C_neg

    normalized_g_pos_list = np.asarray(g_pos_list) / max_value_G_pos
    normalized_g_neg_list = np.asarray(g_neg_list) / max_value_G_neg
    '''
    # This chunk is finding the median values of each position
    median_a_pos_list = position_median(normalized_a_pos_list)
    median_a_neg_list = position_median(normalized_a_neg_list)
    median_t_pos_list = position_median(normalized_t_pos_list)
    median_t_neg_list = position_median(normalized_t_neg_list)
    median_g_pos_list = position_median(normalized_g_pos_list)
    median_g_neg_list = position_median(normalized_g_neg_list)
    median_c_pos_list = position_median(normalized_c_pos_list)
    median_c_neg_list = position_median(normalized_c_neg_list)

    # This part is graphing the medians of every position
    adv_graph_one(median_a_pos_list)
    adv_graph_one(median_a_neg_list)
    adv_graph_one(median_t_pos_list)
    adv_graph_one(median_t_neg_list)
    adv_graph_one(median_g_pos_list)
    adv_graph_one(median_g_neg_list)
    adv_graph_one(median_c_pos_list)
    adv_graph_one(median_c_neg_list)
    '''
    # This chunk is the mean and the graphing of the mean values
    mean_a_pos_list = position_mean(normalized_a_pos_list)
    mean_a_neg_list = position_mean(normalized_a_neg_list)
    mean_t_pos_list = position_mean(normalized_t_pos_list)
    mean_t_neg_list = position_mean(normalized_t_neg_list)
    mean_g_pos_list = position_mean(normalized_g_pos_list)
    mean_g_neg_list = position_mean(normalized_g_neg_list)
    mean_c_pos_list = position_mean(normalized_c_pos_list)
    mean_c_neg_list = position_mean(normalized_c_neg_list)

    #print(mean_a_neg_list)
    adv_graph_one(mean_a_pos_list)
    adv_graph_one(mean_a_neg_list)
    adv_graph_one(mean_t_pos_list)
    adv_graph_one(mean_t_neg_list)
    adv_graph_one(mean_g_pos_list)
    adv_graph_one(mean_g_neg_list)
    adv_graph_one(mean_c_pos_list)
    adv_graph_one(mean_c_neg_list)

    return normalized_a_pos_list


def position_median(nucleotide_list):
    new_list = []
    for position_list in nucleotide_list:
        if len(position_list) == 0:
            new_list.append(0)
        else:
            median_of_list = statistics.median(position_list)
            new_list.append(median_of_list)
    return new_list


def position_mean(nucleotide_list):
    new_list = []
    for position_list in nucleotide_list:
        if len(position_list) == 0:
            new_list.append(0)
        else:
            mean_of_list = statistics.mean(position_list)
            new_list.append(mean_of_list)

    return new_list


def average_pred(data, one_model, target_class):
    data_list = get_collapsed_pred(data, one_model, target_class)
    data_array = np.asarray(data_list)

    # not normalized yet
    mean_data_array = np.mean(data_array, axis=0)

    adv_graph_one(mean_data_array)
    return mean_data_array


def st_dev_pred(data, one_model, target_class):
    data_list = get_collapsed_pred(data, one_model, target_class)
    data_array = np.asarray(data_list)
    std_data_array = np.std(data_array, axis=0)

    adv_graph_one(std_data_array)
    return std_data_array


def median_pred(data, one_model, target_class):
    data_list = get_collapsed_pred(data, one_model, target_class)
    data_array = np.asarray(data_list)
    median_data_array = np.median(data_array, axis=0)

    adv_graph_one(median_data_array)
    return median_data_array


def graph_mean_median_preds(data, one_model, target_class):
    mean_values = average_pred(data, one_model, target_class)
    median_values = median_pred(data, one_model, target_class)

    plot_two_graph(mean_values, median_values, target_class)
#two graphs
#mean for two classes
#median for two classes
#make sure medians and means have same axis /scaling. axis
# graph next to each other / dont overlap graph
# location of TIS + legend

# y axis Adversarial Signals
# matplotlib grid

#ATG of one line in the middle where the highest peak?? draw the line -- this is the location of TIS
#Put legend of that too as well..
#one graph where mean/median for one class - both not overlapping.
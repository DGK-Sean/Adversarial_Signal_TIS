import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def graph_maker(train_list, test_list, epoch_list, model_no):
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(f"accuracy vs epoch of trained and test model {model_no}")

    y1 = train_list
    plt.plot(epoch_list, y1, label="train acc", color="orange")

    y2 = test_list
    plt.plot(epoch_list, y2, label="test acc", color="green")

    plt.legend()
    #plt.show()
    plt.savefig(f"Test Model {model_no}")


def adv_graph(one_big_list):
    list_203 = []
    for i in range(203):
        list_203.append(i)
    plt.xlabel("Nucleotides Position")
    plt.ylabel("Adversarial Signal")

    y1 = one_big_list
    for point in one_big_list:
        plt.plot(list_203, point)

    plt.grid(True, linestyle='--')
    plt.xlim([0, 203])
    plt.ylim([0, 0.20])
    # Vertical Line indicating A-T-G respectively
    plt.axvline(x=60, color='r', linestyle='--', linewidth=1)
    #plt.axvline(x=61, color='r', linestyle='--', linewidth=1)
    #plt.axvline(x=62, color='r', linestyle='--', linewidth=1)

    p_atg = mpatches.Patch(facecolor='red', label='Translation Initiation Location', lw=1, edgecolor='black')
    plt.legend(handles=[p_atg], loc='upper right')
    plt.show()


def adv_graph_one(one_list):
    list_203 = []
    for i in range(203):
        list_203.append(i)
    plt.xlabel("Nucleotides Position")
    plt.ylabel("Adversarial Signal")

    y1 = one_list
    plt.grid(True, linestyle='--')
    plt.plot(list_203, y1)
    plt.xlim([0, 203])
    plt.ylim([-0.20, 0.25])
    plt.axvline(x=60, color='r', linestyle='--', linewidth=1)
    #plt.axvline(x=61, color='r', linestyle='--', linewidth=1)
    #plt.axvline(x=62, color='r', linestyle='--', linewidth=1)

    p_atg = mpatches.Patch(facecolor='red', label='Translation Initiation Location', lw=1, edgecolor='black')
    plt.legend(handles=[p_atg], loc='upper right')
    plt.show()


def plot_two_graph(mean_list, median_list, target_class):
    # Way to plot two graphs side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
    fig.suptitle(f'Mean vs Median of Class {target_class}')
    ax1.set_xlabel("Nucleotides Position")
    ax1.set_ylabel("Adversarial Signal")

    ax2.set_xlabel("Nucleotides Position")

    list_203 = []
    for i in range(203):
        list_203.append(i)
    x = list_203

    # Plotting mean graph
    ax1.plot(x, mean_list, 'tab:green')
    ax1.grid(True, linestyle='--')
    ax1.set_xlim([0, 203])
    ax1.set_ylim([0, 0.20])

    # Plotting median graph
    ax2.plot(x, median_list, 'tab:orange')
    plt.grid(True, linestyle='--')
    ax2.set_xlim([0, 203])
    ax2.set_ylim([0, 0.20])

    # where the TIS is
    ax1.axvline(x=60, color='r', linestyle='--', linewidth=1)
    # ax1.axvline(x=61, color='r', linestyle='--', linewidth=1)
    # ax1.axvline(x=62, color='r', linestyle='--', linewidth=1)

    ax2.axvline(x=60, color='r', linestyle='--', linewidth=1)
    # ax2.axvline(x=61, color='r', linestyle='--', linewidth=1)
    # ax2.axvline(x=62, color='r', linestyle='--', linewidth=1)

    # Legends
    p_median = mpatches.Patch(facecolor='orange', label='Median', lw=1, edgecolor='black')
    p_mean = mpatches.Patch(facecolor='green', label='Mean', lw=1, edgecolor='black')
    p_atg = mpatches.Patch(facecolor='red', label='Translation Initiation Location', lw=1, edgecolor='black')
    plt.legend(handles=[p_mean, p_median, p_atg], loc='upper right')
    #plt.savefig('Adv_Signal_of_test_TIS_data')
    plt.show()
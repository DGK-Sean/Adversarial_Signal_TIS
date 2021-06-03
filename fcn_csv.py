import csv


def csv_writer(list_write):
    with open('model_tracker_edited.csv', 'a', newline='') as f:
        the_writer = csv.writer(f)
        # the_writer.writerow(['model', 'epoch', 'lr', 'momentum', 'nesterov','Batch Size', 'train acc', 'test acc'])
        the_writer.writerow(list_write)


def max_acc_csv(get_list):
    with open('max_model_tracker_edited.csv', 'a', newline='') as f:
        the_writer = csv.writer(f)
        the_writer.writerow(get_list)


def adversarial_csv(get_list):
    with open('adversarial_track.csv', 'a', newline='') as f:
        the_writer = csv.writer(f)
        the_writer.writerow(get_list)
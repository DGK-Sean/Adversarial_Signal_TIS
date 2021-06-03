def find_max(take_list):
    return max([max(sublist) for sublist in take_list if len(sublist) > 0])


def make_absolute(take_list):
    new_list = [[] for _ in range(203)]
    for sublist in take_list:
        if len(sublist) > 0:
            for pos, values in enumerate(sublist):
                new_list[pos].append(abs(values))
    return new_list


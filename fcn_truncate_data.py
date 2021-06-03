from fcn_textfile_divide import text_div


def truncate_data(pos_file, neg_file):
    pos_text = text_div(pos_file)
    neg_text = text_div(neg_file)
    pos_len = len(pos_text)
    neg_len = len(neg_text)
    with open("chrom21_60-140_balanced.neg", "w") as neg_balanced:
        for index in range(pos_len):
            neg_balanced.write(neg_text[index] + '\n')

    # check if they are balanced
    neg_bal = text_div("chrom21_60-140_balanced.neg")
    return len(neg_bal), pos_len, pos_len
#print(truncate_data("../dataset/raw_data/human/train/CCDS60-140.pos", "../dataset/raw_data/human/train/CCDS60-140.neg"))
print(truncate_data("../dataset/raw_data/human/test/chrom21_60-140.pos", "../dataset/raw_data/human/test/chrom21_60-140.neg"))


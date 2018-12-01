#!/usr/bin/python
# --*-- coding: utf-8 --*--


def read_file(file_name_result):
    fr = open(file_name_result, "r")
    print fr.readline()
    data_true = []
    data_false = []
    while 1:
        lines = fr.readline()
        if not lines:
            break
        line = lines.replace('"', '').replace("\n", "").split(",")
        if 0.03 <= float(line[1]) < 0.3:
            data_false.append(line[0])
        if 0.989 <= float(line[1]) <= 0.991:
            data_true.append(line[0])
    fr.close()
    return data_true, data_false


def expand_data(file_name_result, file_data_set, exp_data_file):
    data_true, data_false = read_file(file_name_result)
    print(len(data_true), len(data_false))
    fr = open(file_data_set, "r")
    fopen_data = open(exp_data_file, "w")
    line = fr.readline()
    print line
    # while line:
    while line:
        line = line.replace("\n", "").split(" ")
        if line[0] in data_true:
            line.append('1')
            fopen_data.write(" ".join(line) + "\n")
        if line[0] in data_false:
            line.append('0')
            fopen_data.write(" ".join(line) + "\n")
        line = fr.readline()
    fr.close()
    fopen_data.close()


def main():
    file_name_result = "../result/2017_06_21_24_01.csv"
    file_data_set = "../data/dsjtzs_txfz_test1.txt/dsjtzs_txfz_test1.txt"
    exp_data_file = "../data/dsjtzs_txfz_expand_data.txt"
    expand_data(file_name_result, file_data_set, exp_data_file)


if __name__ == "__main__":
    main()

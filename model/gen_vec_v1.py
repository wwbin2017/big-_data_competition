#!/usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np


def load_file(in_file):
    info = {}
    fr = open(in_file, 'r')
    for line in fr.readlines():
        tmp = line.split()
        tmp[0] = int(tmp[0])
        if tmp[0] not in info:
            info[tmp[0]] = {}
            info[tmp[0]]['points'] = []
            x, y = tmp[2].split(',')
            info[tmp[0]]['target'] = [x, y]
            if len(tmp)>3:
                info[tmp[0]]['label'] = tmp[3]
        for seq in tmp[1].split(';')[:-1]:
            x, y, t = seq.split(',')
            info[tmp[0]]['points'].append([x, y, t])

    fr.close()
    return  info

def gen_train(in_file_train, out_file_train):

    train_data = load_file(in_file_train)
    fw = open(out_file_train, 'w')
    fw.writelines(','.join(['num', 'label',
                            'x_diff_mean_max', 'x_diff_mean_min', 'y_diff_mean_max', 'y_diff_mean_min', 'x_var', 'y_var',
                            'x_resid_mean', 'x_resid_mean_max', 'x_resid_mean_min', 'x_resid_var',
                            'y_resid_mean', 'y_resid_mean_max', 'y_resid_mean_min', 'y_resid_var',
                            't_resid_mean', 't_resid_mean_max', 't_resid_mean_min', 't_resid_var',
                            'xy_dist_mean', 'xy_dist_mean_max', 'xy_dist_mean_min', 'xy_dist_var', 'total_dist',
                            'velo_mean', 'velo_max', 'velo_min', 'velo_var',
                            'p_dist', 't_total', 'point_count'
                            ]) + '\n')
    for k in sorted(train_data.keys()):
        points, target = np.array(train_data[k]['points'], dtype=float), np.array(train_data[k]['target'], dtype=float)

        p_dist = np.linalg.norm(points[0, :2] - target)
        t_total = points[-1, 2] - points[0, 2]

        if points.shape[0] == 1:
            outline = ','.join([str(k), train_data[k]['label']] +
                               [str(0), str(0), str(0), str(0), str(0), str(0)] +
                               [str(-1), str(-1), str(-1), str(-1)] +
                               [str(-1), str(-1), str(-1), str(-1)] +
                               [str(-1), str(-1), str(-1), str(-1)] +
                               [str(-1), str(-1), str(-1), str(-1), str(-1)] +
                               [str(-1), str(-1), str(-1), str(-1)] +
                               [str(round(p_dist, 2)), str(round(t_total, 2)), str(points.shape[0])]
                               ) + '\n'
            fw.writelines(outline)
            continue

        # Step 1
        px, py, pt = points[:, 0], points[:, 1], points[:, 2]
        # print np.mean(px), ',', np.mean(py), ',', np.mean(pt)
        x_diff_mean, y_diff_mean  = px - np.ones(px.shape[0]) * np.mean(px), py - np.ones(py.shape[0]) * np.mean(py)
        x_var, y_var = np.var(px), np.var(py)

        # Step 2
        rx, ry, rt = [], [], []
        for i in range(1, points.shape[0]):
            rx.append(points[i, 0] - points[i - 1, 0])
            ry.append(points[i, 1] - points[i - 1, 1])
            rt.append(points[i, 2] - points[i - 1, 2])
        rx, ry, rt = np.array(rx), np.array(ry), np.array(rt)
        # print rx, ',', ry, ',', rt
        rx_mean, ry_mean, rt_mean = np.mean(rx), np.mean(ry), np.mean(rt)
        x_resid_mean = rx - np.ones(rx.shape[0]) * rx_mean
        y_resid_mean = ry - np.ones(ry.shape[0]) * ry_mean
        t_resid_mean = rt - np.ones(rt.shape[0]) * rt_mean
        rx_var, ry_var, rt_var = np.var(rx), np.var(ry), np.var(rt)

        # Step 3
        xy_dist = []
        for i in range(1, points.shape[0]):
            xy_dist.append(np.linalg.norm(points[i, :2] - points[i-1, :2]))
        xy_dist = np.array(xy_dist)
        xy_dist_mean , xy_dist_var= np.mean(xy_dist), np.var(xy_dist)

        # Step 4
        velo = xy_dist / rt
        velo_mean = np.mean(velo) if not np.isinf(np.mean(velo)) else -2
        velo_var = np.var(velo) if not np.isnan(np.var(velo))  else -3
        max_velo = max(velo) if not np.isinf(max(velo))  else -2
        min_velo = min(velo) if not np.isinf(min(velo))  else -2

        arr = [k, int(train_data[k]['label'])] + \
              [max(x_diff_mean), min(x_diff_mean), max(y_diff_mean), min(y_diff_mean), x_var, y_var] + \
              [rx_mean, max(x_resid_mean), min(x_resid_mean), rx_var] + \
              [ry_mean, max(y_resid_mean), min(y_resid_mean), ry_var] + \
              [rt_mean, max(t_resid_mean), min(t_resid_mean), rt_var] + \
              [xy_dist_mean, max(xy_dist), min(xy_dist), xy_dist_var, sum(xy_dist)] + \
              [velo_mean, max_velo, min_velo, velo_var] + \
              [p_dist, t_total, points.shape[0]]

        outline = ','.join([str(line) for line in arr[:2]] + [str(round(line, 5)) for line in arr[2:]]) + '\n'

        fw.writelines(outline)

    fw.close()

def gen_test(in_file_test, out_file_test):
    train_data = load_file(in_file_test)
    fw = open(out_file_test, 'w')
    fw.writelines(','.join(['num',
                            'x_diff_mean_max', 'x_diff_mean_min', 'y_diff_mean_max', 'y_diff_mean_min', 'x_var', 'y_var',
                            'x_resid_mean', 'x_resid_mean_max', 'x_resid_mean_min', 'x_resid_var',
                            'y_resid_mean', 'y_resid_mean_max', 'y_resid_mean_min', 'y_resid_var',
                            't_resid_mean', 't_resid_mean_max', 't_resid_mean_min', 't_resid_var',
                            'xy_dist_mean', 'xy_dist_mean_max', 'xy_dist_mean_min', 'xy_dist_var', 'total_dist',
                            'velo_mean', 'velo_max', 'velo_min', 'velo_var',
                            'p_dist', 't_total', 'point_count'
                            ]) + '\n')
    for k in sorted(train_data.keys()):
        points, target = np.array(train_data[k]['points'], dtype=float), np.array(train_data[k]['target'], dtype=float)

        p_dist = np.linalg.norm(points[0, :2] - target)
        t_total = points[-1, 2] - points[0, 2]

        if points.shape[0] == 1:
            outline = ','.join([str(k)] +
                               [str(0), str(0), str(0), str(0), str(0), str(0)] +
                               [str(-1), str(-1), str(-1), str(-1)] +
                               [str(-1), str(-1), str(-1), str(-1)] +
                               [str(-1), str(-1), str(-1), str(-1)] +
                               [str(-1), str(-1), str(-1), str(-1), str(-1)] +
                               [str(-1), str(-1), str(-1), str(-1)] +
                               [str(round(p_dist, 2)), str(round(t_total, 2)), str(points.shape[0])]
                               ) + '\n'
            fw.writelines(outline)
            continue

        # Step 1
        px, py, pt = points[:, 0], points[:, 1], points[:, 2]
        # print np.mean(px), ',', np.mean(py), ',', np.mean(pt)
        x_diff_mean, y_diff_mean  = px - np.ones(px.shape[0]) * np.mean(px), py - np.ones(py.shape[0]) * np.mean(py)
        x_var, y_var = np.var(px), np.var(py)

        # Step 2
        rx, ry, rt = [], [], []
        for i in range(1, points.shape[0]):
            rx.append(points[i, 0] - points[i - 1, 0])
            ry.append(points[i, 1] - points[i - 1, 1])
            rt.append(points[i, 2] - points[i - 1, 2])
        rx, ry, rt = np.array(rx), np.array(ry), np.array(rt)
        # print rx, ',', ry, ',', rt
        rx_mean, ry_mean, rt_mean = np.mean(rx), np.mean(ry), np.mean(rt)
        x_resid_mean = rx - np.ones(rx.shape[0]) * rx_mean
        y_resid_mean = ry - np.ones(ry.shape[0]) * ry_mean
        t_resid_mean = rt - np.ones(rt.shape[0]) * rt_mean
        rx_var, ry_var, rt_var = np.var(rx), np.var(ry), np.var(rt)

        # Step 3
        xy_dist = []
        for i in range(1, points.shape[0]):
            xy_dist.append(np.linalg.norm(points[i, :2] - points[i-1, :2]))
        xy_dist = np.array(xy_dist)
        xy_dist_mean , xy_dist_var= np.mean(xy_dist), np.var(xy_dist)

        # Step 4
        velo = xy_dist / rt
        velo_mean = np.mean(velo) if not np.isinf(np.mean(velo)) else -2
        velo_var = np.var(velo) if not np.isnan(np.var(velo))  else -3
        max_velo = max(velo) if not np.isinf(max(velo))  else -2
        min_velo = min(velo) if not np.isinf(min(velo))  else -2

        arr = [k] + \
              [max(x_diff_mean), min(x_diff_mean), max(y_diff_mean), min(y_diff_mean), x_var, y_var] + \
              [rx_mean, max(x_resid_mean), min(x_resid_mean), rx_var] + \
              [ry_mean, max(y_resid_mean), min(y_resid_mean), ry_var] + \
              [rt_mean, max(t_resid_mean), min(t_resid_mean), rt_var] + \
              [xy_dist_mean, max(xy_dist), min(xy_dist), xy_dist_var, sum(xy_dist)] + \
              [velo_mean, max_velo, min_velo, velo_var] + \
              [p_dist, t_total, points.shape[0]]

        outline = ','.join([str(line) for line in arr[:1]] +[str(round(line, 5)) for line in arr[1:]]) + '\n'

        fw.writelines(outline)

    fw.close()

def main():
    in_file_train, in_file_test = '../data/dsjtzs_txfz_training.txt', '../data/dsjtzs_txfz_test1.txt'
    out_file_train, out_file_test = '../tmp/train_vec_v1.csv', '../tmp/test_vec_v1.csv'
    gen_train(in_file_train=in_file_train, out_file_train=out_file_train)
    gen_test(in_file_test=in_file_test, out_file_test=out_file_test)

if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np

def load_file(in_file, is_raw):
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
        x_old, y_old, t_old = -1, -1, -1
        for seq in tmp[1].split(';')[:-1]:
            x, y, t = seq.split(',')
            # if t != t_old and (x != x_old or y != y_old):
            if not is_raw:
                if t != t_old and (x != x_old ):
                    info[tmp[0]]['points'].append([x, y, t])
                    x_old, y_old, t_old = x, y, t
            else:
                info[tmp[0]]['points'].append([x, y, t])

    fr.close()
    return  info

def remove_tzero(dist, t):
    d_info, t_info = [], []
    for i in range(len(t)):
        if t[i] > 0:
            d_info.append(dist[i])
            t_info.append(t[i])

    d_info, t_info = np.array(d_info), np.array(t_info)
    return d_info, t_info

def gen_train(in_file_train, out_file_train):

    train_data = load_file(in_file_train, False)
    train_data_raw = load_file(in_file_train, True)


    fw = open(out_file_train, 'w')
    fw.writelines(','.join(['num', 'label',
                            'x_std', 'y_std', 't_std',
                            'x_resid_mean', 'x_resid_std',
                            'y_resid_mean', 'y_resid_std',
                            't_resid_mean', 't_resid_std',
                            'x_dir_mean',  'x_dir_std',
                            'y_dir_mean', 'y_dir_std',
                            'xy_dir_mean', 'xy_dir_std',
                            'xy_dist_mean', 'xy_dist_std', 'total_dist',
                            'velo_mean', 'velo_std',
                            'x_velo_mean', 'x_velo_std',
                            'y_velo_mean', 'y_velo_std',
                            'x_acc_velo_mean', 'x_acc_velo_std',
                            'y_acc_velo_mean', 'y_acc_velo_std',
                            'xy_acc_velo_mean',  'xy_acc_velo_std',
                            'x_dir_acc_velo_mean', 'x_dir_acc_velo_std',
                            'y_dir_acc_velo_mean', 'y_dir_acc_velo_std',
                            'xy_dir_acc_velo_mean', 'xy_dir_acc_velo_std',
                            'x_dist', 'y_dist', 'p_dist', 't_total', 'point_count',
                            't_before_target', 't_after_target', 't_per',
                            'pc_before_target',  'pc_after_target', 'pc_per'
                            ]) + '\n')


    for k in sorted(train_data.keys()):
        points, target = np.array(train_data[k]['points'], dtype=float), np.array(train_data[k]['target'], dtype=float)
        points_raw = np.array(train_data_raw[k]['points'], dtype=float)

        x_dist, y_dist = np.abs(target[0]-points_raw[-1, 0]), np.abs(target[1]-points_raw[-1, 1])
        p_dist = np.linalg.norm(points_raw[-1, :2] - target)
        t_total = points_raw[-1, 2] - points_raw[0, 2]

        tb, te = points_raw[0, -1], points_raw[-1, -1]
        x, y, t = 0, 0, 0
        c = 0
        for p in points_raw:
            x, y, t = float(p[0]), float(p[1]), float(p[2])
            c += 1
            if x > target[0]:
                break
        t_before_target, t_after_target = t , te - t
        pc_before_target, pc_after_target = c, points_raw.shape[0] - c,
        t_per, pc_per = (t / te) * 100, (c / float(points_raw.shape[0])) * 100

        if points.shape[0] > 1:
            # Step 1 data processing & displacement
            px, py, pt = points[:, 0], points[:, 1], points[:, 2]
            x_diff_mean, y_diff_mean  = px - np.ones(px.shape[0]) * np.mean(px), py - np.ones(py.shape[0]) * np.mean(py)
            x_diff_mean_max, x_diff_mean_min = max(x_diff_mean), min(x_diff_mean)
            y_diff_mean_max, y_diff_mean_min = max(y_diff_mean), min(y_diff_mean)
            x_std, y_std, t_std = np.std(points_raw[:, 0]), np.std(points_raw[:, 1]), np.std(points_raw[:, 2])

            # Step 2 residual
            rx, ry, rt = np.diff(px), np.diff(py), np.diff(pt)
            rx_mean, ry_mean, rt_mean = np.mean(rx), np.mean(ry), np.mean(rt)
            x_resid_max, x_resid_min = max(rx), min(rx)
            y_resid_max, y_resid_min = max(ry), min(ry)
            t_resid_max, t_resid_min = max(rt), min(rt)
            x_resid_mean = (rx - np.ones(rx.shape[0]) * rx_mean)
            y_resid_mean = (ry - np.ones(ry.shape[0]) * ry_mean)
            t_resid_mean = (rt - np.ones(rt.shape[0]) * rt_mean)
            x_resid_mean_max, x_resid_mean_min = max(x_resid_mean), min(x_resid_mean)
            y_resid_mean_max, y_resid_mean_min = max(y_resid_mean), min(y_resid_mean)
            t_resid_mean_max, t_resid_mean_min = max(t_resid_mean), min(t_resid_mean)
            rx_std, ry_std, rt_std = np.std(rx), np.std(ry), np.std(rt)

            # Step 3
            xy_unit = []
            for i in range(rx.shape[0]):
                xy_unit.append(np.linalg.norm(np.array([rx[i],ry[i]])))
            xy_unit = np.array(xy_unit)
            x_dir, y_dir, xy_dir = rx / xy_unit, ry / xy_unit, np.arctan(ry / rx)
            x_dir_mean, y_dir_mean, xy_dir_mean = np.mean(x_dir), np.mean(y_dir), np.mean(xy_dir)
            x_dir_max, y_dir_max, xy_dir_max = max(x_dir), max(y_dir), max(xy_dir)
            x_dir_min, y_dir_min, xy_dir_min = min(x_dir), min(y_dir), min(xy_dir)
            dx_mean = x_dir - np.ones(x_dir.shape[0]) * x_dir_mean
            dy_mean =  y_dir - np.ones(y_dir.shape[0]) * y_dir_mean
            dxy_mean = xy_dir - np.ones(xy_dir.shape[0]) * xy_dir_mean
            x_dir_mean_max, y_dir_mean_max, xy_dir_mean_max = max(dx_mean), max(dy_mean), max(dxy_mean)
            x_dir_mean_min, y_dir_mean_min, xy_dir_mean_min = min(dx_mean), min(dy_mean), min(dxy_mean)
            x_dir_std, y_dir_std, xy_dir_std = np.std(x_dir), np.std(y_dir), np.std(xy_dir)

            xy_dist = []
            for i in range(1, points.shape[0]):
                xy_dist.append(np.linalg.norm(points[i, :2] - points[i-1, :2]))
            xy_dist = np.array(xy_dist)
            xy_dist_max, xy_dist_min = max(xy_dist), min(xy_dist)
            xy_dist_mean , xy_dist_std= np.mean(xy_dist), np.std(xy_dist)
            dxy_dist_mean = xy_dist - np.ones(xy_dist.shape[0]) * xy_dist_mean
            xy_dist_mean_max, xy_dist_mean_min = max(dxy_dist_mean), min(dxy_dist_mean)
            xy_dist_sum = sum(xy_dist)

            # Step 4 velocity
            velo = np.abs(xy_dist) / rt
            velo_mean, velo_std = np.mean(velo), np.std(velo)
            max_velo, min_velo = max(velo), min(velo)
            dvelo_mean = velo - np.ones(velo.shape[0]) * velo_mean
            max_velo_mean, min_velo_mean = max(dvelo_mean), min(dvelo_mean)

            x_velo = np.abs(rx) /rt
            x_velo_mean, x_velo_std = np.mean(x_velo), np.std(x_velo)
            max_x_velo, min_x_velo = max(x_velo), min(x_velo)
            dx_velo_mean = x_velo - np.ones(x_velo.shape[0]) * x_velo_mean
            max_x_velo_mean, min_x_velo_mean = max(dx_velo_mean), min(dx_velo_mean)

            y_velo = np.abs(ry) /rt
            y_velo_mean, y_velo_std = np.mean(y_velo), np.std(y_velo)
            max_y_velo, min_y_velo = max(y_velo), min(y_velo)
            dy_velo_mean = y_velo - np.ones(y_velo.shape[0]) * y_velo_mean
            max_y_velo_mean, min_y_velo_mean = max(dy_velo_mean), min(dy_velo_mean)

            if points.shape[0] > 2:
                # Step 5 accelerated velocity
                x_acc_velo, y_acc_velo, xy_acc_velo = np.abs(np.diff(x_velo))/rt[1:], np.abs(np.diff(y_velo))/rt[1:], np.abs(np.diff(velo))/rt[1:]
                x_acc_velo_mean, y_acc_velo_mean, xy_acc_velo_mean = np.mean(x_acc_velo), np.mean(y_acc_velo), np.mean(xy_acc_velo)
                x_acc_velo_std, y_acc_velo_std, xy_acc_velo_std = np.std(x_acc_velo), np.std(y_acc_velo), np.std(xy_acc_velo)
                x_acc_velo_max, y_acc_velo_max, xy_acc_velo_max = max(x_acc_velo), max(y_acc_velo), max(xy_acc_velo)
                x_acc_velo_min, y_acc_velo_min, xy_acc_velo_min = min(x_acc_velo), min(y_acc_velo), min(xy_acc_velo)
                dx_acc_velo = x_acc_velo - np.ones(x_acc_velo.shape[0]) * x_acc_velo_mean
                dy_acc_velo = y_acc_velo - np.ones(y_acc_velo.shape[0]) * y_acc_velo_mean
                dxy_acc_velo = xy_acc_velo - np.ones(xy_acc_velo.shape[0]) * xy_acc_velo_mean
                x_acc_velo_mean_max, y_acc_velo_mean_max, xy_acc_velo_mean_max = max(dx_acc_velo), max(dy_acc_velo), max(dxy_acc_velo)
                x_acc_velo_mean_min, y_acc_velo_mean_min, xy_acc_velo_mean_min = min(dx_acc_velo), min(dy_acc_velo), min(dxy_acc_velo)

                x_dir_acc_velo, y_dir_acc_velo, xy_dir_acc_velo = np.abs(np.diff(x_dir))/rt[1:], np.abs(np.diff(y_dir))/rt[1:], np.abs(np.diff(xy_dir))/rt[1:]
                x_dir_acc_velo_mean, y_dir_acc_velo_mean, xy_dir_acc_velo_mean = np.mean(x_dir_acc_velo), np.mean(y_dir_acc_velo), np.mean(xy_dir_acc_velo)
                x_dir_acc_velo_std, y_dir_acc_velo_std, xy_dir_acc_velo_std = np.std(x_dir_acc_velo), np.std(y_dir_acc_velo), np.std(xy_dir_acc_velo)
                x_dir_acc_velo_max, y_dir_acc_velo_max, xy_dir_acc_velo_max = max(x_dir_acc_velo), max(y_dir_acc_velo), max(xy_dir_acc_velo)
                x_dir_acc_velo_min, y_dir_acc_velo_min, xy_dir_acc_velo_min = min(x_dir_acc_velo), min(y_dir_acc_velo), max(xy_dir_acc_velo)
                dx_dir_acc_velo = x_dir_acc_velo - np.ones(x_dir_acc_velo.shape[0]) * x_dir_acc_velo_mean
                dy_dir_acc_velo = y_dir_acc_velo - np.ones(y_dir_acc_velo.shape[0]) * y_dir_acc_velo_mean
                dxy_dir_acc_velo = xy_dir_acc_velo - np.ones(xy_dir_acc_velo.shape[0]) * xy_dir_acc_velo_mean
                x_dir_acc_velo_mean_max, y_dir_acc_velo_mean_max, xy_dir_acc_velo_mean_max = max(dx_dir_acc_velo), max(dy_dir_acc_velo), max(dxy_dir_acc_velo)
                x_dir_acc_velo_mean_min, y_dir_acc_velo_mean_min, xy_dir_acc_velo_mean_min = min(dx_dir_acc_velo), min(dy_dir_acc_velo), min(dxy_dir_acc_velo)

            else:
                x_acc_velo_mean, x_acc_velo_max, x_acc_velo_min, x_acc_velo_std = -1, -1, -1, -1
                y_acc_velo_mean, y_acc_velo_max, y_acc_velo_min, y_acc_velo_std = -1, -1, -1, -1
                xy_acc_velo_mean, xy_acc_velo_max, xy_acc_velo_min, xy_acc_velo_std = -1, -1, -1, -1
                x_acc_velo_mean_max, y_acc_velo_mean_max, xy_acc_velo_mean_max = -1, -1, -1
                x_acc_velo_mean_min, y_acc_velo_mean_min, xy_acc_velo_mean_min = -1, -1, -1
                x_dir_acc_velo_mean, x_dir_acc_velo_max, x_dir_acc_velo_min, x_dir_acc_velo_std = -1, -1, -1, -1
                y_dir_acc_velo_mean, y_dir_acc_velo_max, y_dir_acc_velo_min, y_dir_acc_velo_std = -1, -1, -1, -1
                xy_dir_acc_velo_mean, xy_dir_acc_velo_max, xy_dir_acc_velo_min, xy_dir_acc_velo_std = -1, -1, -1, -1
                x_dir_acc_velo_mean_max, y_dir_acc_velo_mean_max, xy_dir_acc_velo_mean_max = -1, -1, -1
                x_dir_acc_velo_mean_min, y_dir_acc_velo_mean_min, xy_dir_acc_velo_mean_min = -1, -1, -1
        else:
            x_diff_mean_max, x_diff_mean_min, y_diff_mean_max, y_diff_mean_min, x_std, y_std = -1, -1, -1, -1, -1, -1
            rx_mean, x_resid_max, x_resid_min, x_resid_mean_max, x_resid_mean_min, rx_std = -1, -1, -1, -1, -1, -1
            ry_mean, y_resid_max, y_resid_min, y_resid_mean_max, y_resid_mean_min, ry_std = -1, -1, -1, -1, -1, -1
            rt_mean, t_resid_max, t_resid_min, t_resid_mean_max, t_resid_mean_min, rt_std = -1, -1, -1, -1, -1, -1
            x_dir_mean, x_dir_max, x_dir_min, x_dir_std = -1, -1, -1, -1
            y_dir_mean, y_dir_max, y_dir_min, y_dir_std = -1, -1, -1, -1
            xy_dir_mean, xy_dir_max, xy_dir_min, xy_dir_std = -1, -1, -1, -1
            x_dir_mean_max, y_dir_mean_max, xy_dir_mean_max = -1, -1, -1
            x_dir_mean_min, y_dir_mean_min, xy_dir_mean_min = -1, -1, -1
            xy_dist_mean, xy_dist_max, xy_dist_min, xy_dist_std, xy_dist_sum = -1, -1, -1, -1, -1
            xy_dist_mean_max, xy_dist_mean_min = -1, -1
            velo_mean, max_velo, min_velo, velo_std = -1, -1, -1, -1
            max_velo_mean, min_velo_mean = -1, -1
            x_velo_mean, max_x_velo, min_x_velo, x_velo_std = -1, -1, -1, -1
            max_x_velo_mean, min_x_velo_mean = -1, -1
            y_velo_mean, max_y_velo, min_y_velo, y_velo_std = -1, -1, -1, -1
            max_y_velo_mean, min_y_velo_mean = -1, -1
            x_acc_velo_mean, x_acc_velo_max, x_acc_velo_min, x_acc_velo_std = -1, -1, -1, -1
            y_acc_velo_mean, y_acc_velo_max, y_acc_velo_min, y_acc_velo_std = -1, -1, -1, -1
            xy_acc_velo_mean, xy_acc_velo_max, xy_acc_velo_min, xy_acc_velo_std = -1, -1, -1, -1
            x_dir_acc_velo_mean, x_dir_acc_velo_max, x_dir_acc_velo_min, x_dir_acc_velo_std = -1, -1, -1, -1
            y_dir_acc_velo_mean, y_dir_acc_velo_max, y_dir_acc_velo_min, y_dir_acc_velo_std = -1, -1, -1, -1
            xy_dir_acc_velo_mean, xy_dir_acc_velo_max, xy_dir_acc_velo_min, xy_dir_acc_velo_std = -1, -1, -1, -1

        arr = [k, int(train_data[k]['label'])] + \
              [x_std, y_std] + \
              [rx_mean, rx_std] + \
              [ry_mean, ry_std] + \
              [rt_mean, rt_std] + \
              [x_dir_mean, x_dir_std] + \
              [y_dir_mean, y_dir_std] + \
              [xy_dir_mean, xy_dir_std] + \
              [xy_dist_mean, xy_dist_std, xy_dist_sum] + \
              [velo_mean, velo_std] + \
              [x_velo_mean, x_velo_std] + \
              [y_velo_mean, y_velo_std] + \
              [x_acc_velo_mean, x_acc_velo_std] + \
              [y_acc_velo_mean, y_acc_velo_std] + \
              [xy_acc_velo_mean, xy_acc_velo_std] + \
              [x_dir_acc_velo_mean, x_dir_acc_velo_std] + \
              [y_dir_acc_velo_mean, y_dir_acc_velo_std] + \
              [xy_dir_acc_velo_mean, xy_dir_acc_velo_std] + \
              [x_dist, y_dist, p_dist, t_total, points_raw.shape[0]] + \
              [t_before_target, t_after_target, t_per] + \
              [pc_before_target, pc_after_target, pc_per]

        outline = ','.join([str(line) for line in arr[:2]] + [str(round(line, 5)) for line in arr[2:]]) + '\n'

        fw.writelines(outline)

    fw.close()

def gen_train_expand(in_file_train, out_file_train):

    train_data = load_file(in_file_train, False)
    train_data_raw = load_file(in_file_train, True)


    fw = open(out_file_train, 'a')
    '''
    fw.writelines(','.join(['num', 'label',
                            'x_std', 'y_std', 't_std',
                            'x_resid_mean', 'x_resid_std',
                            'y_resid_mean', 'y_resid_std',
                            't_resid_mean', 't_resid_std',
                            'x_dir_mean',  'x_dir_std',
                            'y_dir_mean', 'y_dir_std',
                            'xy_dir_mean', 'xy_dir_std',
                            'xy_dist_mean', 'xy_dist_std', 'total_dist',
                            'velo_mean', 'velo_std',
                            'x_velo_mean', 'x_velo_std',
                            'y_velo_mean', 'y_velo_std',
                            'x_acc_velo_mean', 'x_acc_velo_std',
                            'y_acc_velo_mean', 'y_acc_velo_std',
                            'xy_acc_velo_mean',  'xy_acc_velo_std',
                            'x_dir_acc_velo_mean', 'x_dir_acc_velo_std',
                            'y_dir_acc_velo_mean', 'y_dir_acc_velo_std',
                            'xy_dir_acc_velo_mean', 'xy_dir_acc_velo_std',
                            'x_dist', 'y_dist', 'p_dist', 't_total', 'point_count',
                            't_before_target', 't_after_target', 't_per',
                            'pc_before_target',  'pc_after_target', 'pc_per'
                            ]) + '\n')

    '''
    for k in sorted(train_data.keys()):
        points, target = np.array(train_data[k]['points'], dtype=float), np.array(train_data[k]['target'], dtype=float)
        points_raw = np.array(train_data_raw[k]['points'], dtype=float)

        x_dist, y_dist = np.abs(target[0]-points_raw[-1, 0]), np.abs(target[1]-points_raw[-1, 1])
        p_dist = np.linalg.norm(points_raw[-1, :2] - target)
        t_total = points_raw[-1, 2] - points_raw[0, 2]

        tb, te = points_raw[0, -1], points_raw[-1, -1]
        x, y, t = 0, 0, 0
        c = 0
        for p in points_raw:
            x, y, t = float(p[0]), float(p[1]), float(p[2])
            c += 1
            if x > target[0]:
                break
        t_before_target, t_after_target = t , te - t
        pc_before_target, pc_after_target = c, points_raw.shape[0] - c,
        t_per, pc_per = (t / te) * 100, (c / float(points_raw.shape[0])) * 100

        if points.shape[0] > 1:
            # Step 1 data processing & displacement
            px, py, pt = points[:, 0], points[:, 1], points[:, 2]
            x_diff_mean, y_diff_mean  = px - np.ones(px.shape[0]) * np.mean(px), py - np.ones(py.shape[0]) * np.mean(py)
            x_diff_mean_max, x_diff_mean_min = max(x_diff_mean), min(x_diff_mean)
            y_diff_mean_max, y_diff_mean_min = max(y_diff_mean), min(y_diff_mean)
            x_std, y_std, t_std = np.std(points_raw[:, 0]), np.std(points_raw[:, 1]), np.std(points_raw[:, 2])

            # Step 2 residual
            rx, ry, rt = np.diff(px), np.diff(py), np.diff(pt)
            rx_mean, ry_mean, rt_mean = np.mean(rx), np.mean(ry), np.mean(rt)
            x_resid_max, x_resid_min = max(rx), min(rx)
            y_resid_max, y_resid_min = max(ry), min(ry)
            t_resid_max, t_resid_min = max(rt), min(rt)
            x_resid_mean = (rx - np.ones(rx.shape[0]) * rx_mean)
            y_resid_mean = (ry - np.ones(ry.shape[0]) * ry_mean)
            t_resid_mean = (rt - np.ones(rt.shape[0]) * rt_mean)
            x_resid_mean_max, x_resid_mean_min = max(x_resid_mean), min(x_resid_mean)
            y_resid_mean_max, y_resid_mean_min = max(y_resid_mean), min(y_resid_mean)
            t_resid_mean_max, t_resid_mean_min = max(t_resid_mean), min(t_resid_mean)
            rx_std, ry_std, rt_std = np.std(rx), np.std(ry), np.std(rt)

            # Step 3
            xy_unit = []
            for i in range(rx.shape[0]):
                xy_unit.append(np.linalg.norm(np.array([rx[i],ry[i]])))
            xy_unit = np.array(xy_unit)
            x_dir, y_dir, xy_dir = rx / xy_unit, ry / xy_unit, np.arctan(ry / rx)
            x_dir_mean, y_dir_mean, xy_dir_mean = np.mean(x_dir), np.mean(y_dir), np.mean(xy_dir)
            x_dir_max, y_dir_max, xy_dir_max = max(x_dir), max(y_dir), max(xy_dir)
            x_dir_min, y_dir_min, xy_dir_min = min(x_dir), min(y_dir), min(xy_dir)
            dx_mean = x_dir - np.ones(x_dir.shape[0]) * x_dir_mean
            dy_mean =  y_dir - np.ones(y_dir.shape[0]) * y_dir_mean
            dxy_mean = xy_dir - np.ones(xy_dir.shape[0]) * xy_dir_mean
            x_dir_mean_max, y_dir_mean_max, xy_dir_mean_max = max(dx_mean), max(dy_mean), max(dxy_mean)
            x_dir_mean_min, y_dir_mean_min, xy_dir_mean_min = min(dx_mean), min(dy_mean), min(dxy_mean)
            x_dir_std, y_dir_std, xy_dir_std = np.std(x_dir), np.std(y_dir), np.std(xy_dir)

            xy_dist = []
            for i in range(1, points.shape[0]):
                xy_dist.append(np.linalg.norm(points[i, :2] - points[i-1, :2]))
            xy_dist = np.array(xy_dist)
            xy_dist_max, xy_dist_min = max(xy_dist), min(xy_dist)
            xy_dist_mean , xy_dist_std= np.mean(xy_dist), np.std(xy_dist)
            dxy_dist_mean = xy_dist - np.ones(xy_dist.shape[0]) * xy_dist_mean
            xy_dist_mean_max, xy_dist_mean_min = max(dxy_dist_mean), min(dxy_dist_mean)
            xy_dist_sum = sum(xy_dist)

            # Step 4 velocity
            velo = np.abs(xy_dist) / rt
            velo_mean, velo_std = np.mean(velo), np.std(velo)
            max_velo, min_velo = max(velo), min(velo)
            dvelo_mean = velo - np.ones(velo.shape[0]) * velo_mean
            max_velo_mean, min_velo_mean = max(dvelo_mean), min(dvelo_mean)

            x_velo = np.abs(rx) /rt
            x_velo_mean, x_velo_std = np.mean(x_velo), np.std(x_velo)
            max_x_velo, min_x_velo = max(x_velo), min(x_velo)
            dx_velo_mean = x_velo - np.ones(x_velo.shape[0]) * x_velo_mean
            max_x_velo_mean, min_x_velo_mean = max(dx_velo_mean), min(dx_velo_mean)

            y_velo = np.abs(ry) /rt
            y_velo_mean, y_velo_std = np.mean(y_velo), np.std(y_velo)
            max_y_velo, min_y_velo = max(y_velo), min(y_velo)
            dy_velo_mean = y_velo - np.ones(y_velo.shape[0]) * y_velo_mean
            max_y_velo_mean, min_y_velo_mean = max(dy_velo_mean), min(dy_velo_mean)

            if points.shape[0] > 2:
                # Step 5 accelerated velocity
                x_acc_velo, y_acc_velo, xy_acc_velo = np.abs(np.diff(x_velo))/rt[1:], np.abs(np.diff(y_velo))/rt[1:], np.abs(np.diff(velo))/rt[1:]
                x_acc_velo_mean, y_acc_velo_mean, xy_acc_velo_mean = np.mean(x_acc_velo), np.mean(y_acc_velo), np.mean(xy_acc_velo)
                x_acc_velo_std, y_acc_velo_std, xy_acc_velo_std = np.std(x_acc_velo), np.std(y_acc_velo), np.std(xy_acc_velo)
                x_acc_velo_max, y_acc_velo_max, xy_acc_velo_max = max(x_acc_velo), max(y_acc_velo), max(xy_acc_velo)
                x_acc_velo_min, y_acc_velo_min, xy_acc_velo_min = min(x_acc_velo), min(y_acc_velo), min(xy_acc_velo)
                dx_acc_velo = x_acc_velo - np.ones(x_acc_velo.shape[0]) * x_acc_velo_mean
                dy_acc_velo = y_acc_velo - np.ones(y_acc_velo.shape[0]) * y_acc_velo_mean
                dxy_acc_velo = xy_acc_velo - np.ones(xy_acc_velo.shape[0]) * xy_acc_velo_mean
                x_acc_velo_mean_max, y_acc_velo_mean_max, xy_acc_velo_mean_max = max(dx_acc_velo), max(dy_acc_velo), max(dxy_acc_velo)
                x_acc_velo_mean_min, y_acc_velo_mean_min, xy_acc_velo_mean_min = min(dx_acc_velo), min(dy_acc_velo), min(dxy_acc_velo)

                x_dir_acc_velo, y_dir_acc_velo, xy_dir_acc_velo = np.abs(np.diff(x_dir))/rt[1:], np.abs(np.diff(y_dir))/rt[1:], np.abs(np.diff(xy_dir))/rt[1:]
                x_dir_acc_velo_mean, y_dir_acc_velo_mean, xy_dir_acc_velo_mean = np.mean(x_dir_acc_velo), np.mean(y_dir_acc_velo), np.mean(xy_dir_acc_velo)
                x_dir_acc_velo_std, y_dir_acc_velo_std, xy_dir_acc_velo_std = np.std(x_dir_acc_velo), np.std(y_dir_acc_velo), np.std(xy_dir_acc_velo)
                x_dir_acc_velo_max, y_dir_acc_velo_max, xy_dir_acc_velo_max = max(x_dir_acc_velo), max(y_dir_acc_velo), max(xy_dir_acc_velo)
                x_dir_acc_velo_min, y_dir_acc_velo_min, xy_dir_acc_velo_min = min(x_dir_acc_velo), min(y_dir_acc_velo), max(xy_dir_acc_velo)
                dx_dir_acc_velo = x_dir_acc_velo - np.ones(x_dir_acc_velo.shape[0]) * x_dir_acc_velo_mean
                dy_dir_acc_velo = y_dir_acc_velo - np.ones(y_dir_acc_velo.shape[0]) * y_dir_acc_velo_mean
                dxy_dir_acc_velo = xy_dir_acc_velo - np.ones(xy_dir_acc_velo.shape[0]) * xy_dir_acc_velo_mean
                x_dir_acc_velo_mean_max, y_dir_acc_velo_mean_max, xy_dir_acc_velo_mean_max = max(dx_dir_acc_velo), max(dy_dir_acc_velo), max(dxy_dir_acc_velo)
                x_dir_acc_velo_mean_min, y_dir_acc_velo_mean_min, xy_dir_acc_velo_mean_min = min(dx_dir_acc_velo), min(dy_dir_acc_velo), min(dxy_dir_acc_velo)

            else:
                x_acc_velo_mean, x_acc_velo_max, x_acc_velo_min, x_acc_velo_std = -1, -1, -1, -1
                y_acc_velo_mean, y_acc_velo_max, y_acc_velo_min, y_acc_velo_std = -1, -1, -1, -1
                xy_acc_velo_mean, xy_acc_velo_max, xy_acc_velo_min, xy_acc_velo_std = -1, -1, -1, -1
                x_acc_velo_mean_max, y_acc_velo_mean_max, xy_acc_velo_mean_max = -1, -1, -1
                x_acc_velo_mean_min, y_acc_velo_mean_min, xy_acc_velo_mean_min = -1, -1, -1
                x_dir_acc_velo_mean, x_dir_acc_velo_max, x_dir_acc_velo_min, x_dir_acc_velo_std = -1, -1, -1, -1
                y_dir_acc_velo_mean, y_dir_acc_velo_max, y_dir_acc_velo_min, y_dir_acc_velo_std = -1, -1, -1, -1
                xy_dir_acc_velo_mean, xy_dir_acc_velo_max, xy_dir_acc_velo_min, xy_dir_acc_velo_std = -1, -1, -1, -1
                x_dir_acc_velo_mean_max, y_dir_acc_velo_mean_max, xy_dir_acc_velo_mean_max = -1, -1, -1
                x_dir_acc_velo_mean_min, y_dir_acc_velo_mean_min, xy_dir_acc_velo_mean_min = -1, -1, -1
        else:
            x_diff_mean_max, x_diff_mean_min, y_diff_mean_max, y_diff_mean_min, x_std, y_std = -1, -1, -1, -1, -1, -1
            rx_mean, x_resid_max, x_resid_min, x_resid_mean_max, x_resid_mean_min, rx_std = -1, -1, -1, -1, -1, -1
            ry_mean, y_resid_max, y_resid_min, y_resid_mean_max, y_resid_mean_min, ry_std = -1, -1, -1, -1, -1, -1
            rt_mean, t_resid_max, t_resid_min, t_resid_mean_max, t_resid_mean_min, rt_std = -1, -1, -1, -1, -1, -1
            x_dir_mean, x_dir_max, x_dir_min, x_dir_std = -1, -1, -1, -1
            y_dir_mean, y_dir_max, y_dir_min, y_dir_std = -1, -1, -1, -1
            xy_dir_mean, xy_dir_max, xy_dir_min, xy_dir_std = -1, -1, -1, -1
            x_dir_mean_max, y_dir_mean_max, xy_dir_mean_max = -1, -1, -1
            x_dir_mean_min, y_dir_mean_min, xy_dir_mean_min = -1, -1, -1
            xy_dist_mean, xy_dist_max, xy_dist_min, xy_dist_std, xy_dist_sum = -1, -1, -1, -1, -1
            xy_dist_mean_max, xy_dist_mean_min = -1, -1
            velo_mean, max_velo, min_velo, velo_std = -1, -1, -1, -1
            max_velo_mean, min_velo_mean = -1, -1
            x_velo_mean, max_x_velo, min_x_velo, x_velo_std = -1, -1, -1, -1
            max_x_velo_mean, min_x_velo_mean = -1, -1
            y_velo_mean, max_y_velo, min_y_velo, y_velo_std = -1, -1, -1, -1
            max_y_velo_mean, min_y_velo_mean = -1, -1
            x_acc_velo_mean, x_acc_velo_max, x_acc_velo_min, x_acc_velo_std = -1, -1, -1, -1
            y_acc_velo_mean, y_acc_velo_max, y_acc_velo_min, y_acc_velo_std = -1, -1, -1, -1
            xy_acc_velo_mean, xy_acc_velo_max, xy_acc_velo_min, xy_acc_velo_std = -1, -1, -1, -1
            x_dir_acc_velo_mean, x_dir_acc_velo_max, x_dir_acc_velo_min, x_dir_acc_velo_std = -1, -1, -1, -1
            y_dir_acc_velo_mean, y_dir_acc_velo_max, y_dir_acc_velo_min, y_dir_acc_velo_std = -1, -1, -1, -1
            xy_dir_acc_velo_mean, xy_dir_acc_velo_max, xy_dir_acc_velo_min, xy_dir_acc_velo_std = -1, -1, -1, -1

        arr = [k, int(train_data[k]['label'])] + \
              [x_std, y_std] + \
              [rx_mean, rx_std] + \
              [ry_mean, ry_std] + \
              [rt_mean, rt_std] + \
              [x_dir_mean, x_dir_std] + \
              [y_dir_mean, y_dir_std] + \
              [xy_dir_mean, xy_dir_std] + \
              [xy_dist_mean, xy_dist_std, xy_dist_sum] + \
              [velo_mean, velo_std] + \
              [x_velo_mean, x_velo_std] + \
              [y_velo_mean, y_velo_std] + \
              [x_acc_velo_mean, x_acc_velo_std] + \
              [y_acc_velo_mean, y_acc_velo_std] + \
              [xy_acc_velo_mean, xy_acc_velo_std] + \
              [x_dir_acc_velo_mean, x_dir_acc_velo_std] + \
              [y_dir_acc_velo_mean, y_dir_acc_velo_std] + \
              [xy_dir_acc_velo_mean, xy_dir_acc_velo_std] + \
              [x_dist, y_dist, p_dist, t_total, points_raw.shape[0]] + \
              [t_before_target, t_after_target, t_per] + \
              [pc_before_target, pc_after_target, pc_per]

        outline = ','.join([str(line) for line in arr[:2]] + [str(round(line, 5)) for line in arr[2:]]) + '\n'

        fw.writelines(outline)

    fw.close()

def gen_test(in_file_test, out_file_test):

    test_data = load_file(in_file_test, False)
    test_data_raw = load_file(in_file_test, True)


    fw = open(out_file_test, 'w')
    fw.writelines(','.join(['num',
                            'x_std', 'y_std', 't_std',
                            'x_resid_mean', 'x_resid_std',
                            'y_resid_mean', 'y_resid_std',
                            't_resid_mean', 't_resid_std',
                            'x_dir_mean',  'x_dir_std',
                            'y_dir_mean', 'y_dir_std',
                            'xy_dir_mean', 'xy_dir_std',
                            'xy_dist_mean', 'xy_dist_std', 'total_dist',
                            'velo_mean', 'velo_std',
                            'x_velo_mean', 'x_velo_std',
                            'y_velo_mean', 'y_velo_std',
                            'x_acc_velo_mean', 'x_acc_velo_std',
                            'y_acc_velo_mean', 'y_acc_velo_std',
                            'xy_acc_velo_mean',  'xy_acc_velo_std',
                            'x_dir_acc_velo_mean', 'x_dir_acc_velo_std',
                            'y_dir_acc_velo_mean', 'y_dir_acc_velo_std',
                            'xy_dir_acc_velo_mean', 'xy_dir_acc_velo_std',
                            'x_dist', 'y_dist', 'p_dist', 't_total', 'point_count',
                            't_before_target', 't_after_target', 't_per',
                            'pc_before_target',  'pc_after_target', 'pc_per'
                            ]) + '\n')


    for k in sorted(test_data.keys()):
        points, target = np.array(test_data[k]['points'], dtype=float), np.array(test_data[k]['target'], dtype=float)
        points_raw = np.array(test_data_raw[k]['points'], dtype=float)

        x_dist, y_dist = np.abs(target[0]-points_raw[-1, 0]), np.abs(target[1]-points_raw[-1, 1])
        p_dist = np.linalg.norm(points_raw[-1, :2] - target)
        t_total = points_raw[-1, 2] - points_raw[0, 2]

        tb, te = points_raw[0, -1], points_raw[-1, -1]
        x, y, t = 0, 0, 0
        c = 0
        for p in points_raw:
            x, y, t = float(p[0]), float(p[1]), float(p[2])
            c += 1
            if x > target[0]:
                break
        t_before_target, t_after_target = t , te - t
        pc_before_target, pc_after_target = c, points_raw.shape[0] - c,
        t_per, pc_per = (t / te) * 100, (c / float(points_raw.shape[0])) * 100

        if points.shape[0] > 1:
            # Step 1 data processing & displacement
            px, py, pt = points[:, 0], points[:, 1], points[:, 2]
            x_diff_mean, y_diff_mean  = px - np.ones(px.shape[0]) * np.mean(px), py - np.ones(py.shape[0]) * np.mean(py)
            x_diff_mean_max, x_diff_mean_min = max(x_diff_mean), min(x_diff_mean)
            y_diff_mean_max, y_diff_mean_min = max(y_diff_mean), min(y_diff_mean)
            x_std, y_std, t_std = np.std(points_raw[:, 0]), np.std(points_raw[:, 1]), np.std(points_raw[:, 2])

            # Step 2 residual
            rx, ry, rt = np.diff(px), np.diff(py), np.diff(pt)
            rx_mean, ry_mean, rt_mean = np.mean(rx), np.mean(ry), np.mean(rt)
            x_resid_max, x_resid_min = max(rx), min(rx)
            y_resid_max, y_resid_min = max(ry), min(ry)
            t_resid_max, t_resid_min = max(rt), min(rt)
            x_resid_mean = (rx - np.ones(rx.shape[0]) * rx_mean)
            y_resid_mean = (ry - np.ones(ry.shape[0]) * ry_mean)
            t_resid_mean = (rt - np.ones(rt.shape[0]) * rt_mean)
            x_resid_mean_max, x_resid_mean_min = max(x_resid_mean), min(x_resid_mean)
            y_resid_mean_max, y_resid_mean_min = max(y_resid_mean), min(y_resid_mean)
            t_resid_mean_max, t_resid_mean_min = max(t_resid_mean), min(t_resid_mean)
            rx_std, ry_std, rt_std = np.std(rx), np.std(ry), np.std(rt)

            # Step 3
            xy_unit = []
            for i in range(rx.shape[0]):
                xy_unit.append(np.linalg.norm(np.array([rx[i],ry[i]])))
            xy_unit = np.array(xy_unit)
            x_dir, y_dir, xy_dir = rx / xy_unit, ry / xy_unit, np.arctan(ry / rx)
            x_dir_mean, y_dir_mean, xy_dir_mean = np.mean(x_dir), np.mean(y_dir), np.mean(xy_dir)
            x_dir_max, y_dir_max, xy_dir_max = max(x_dir), max(y_dir), max(xy_dir)
            x_dir_min, y_dir_min, xy_dir_min = min(x_dir), min(y_dir), min(xy_dir)
            dx_mean = x_dir - np.ones(x_dir.shape[0]) * x_dir_mean
            dy_mean =  y_dir - np.ones(y_dir.shape[0]) * y_dir_mean
            dxy_mean = xy_dir - np.ones(xy_dir.shape[0]) * xy_dir_mean
            x_dir_mean_max, y_dir_mean_max, xy_dir_mean_max = max(dx_mean), max(dy_mean), max(dxy_mean)
            x_dir_mean_min, y_dir_mean_min, xy_dir_mean_min = min(dx_mean), min(dy_mean), min(dxy_mean)
            x_dir_std, y_dir_std, xy_dir_std = np.std(x_dir), np.std(y_dir), np.std(xy_dir)

            xy_dist = []
            for i in range(1, points.shape[0]):
                xy_dist.append(np.linalg.norm(points[i, :2] - points[i-1, :2]))
            xy_dist = np.array(xy_dist)
            xy_dist_max, xy_dist_min = max(xy_dist), min(xy_dist)
            xy_dist_mean , xy_dist_std= np.mean(xy_dist), np.std(xy_dist)
            dxy_dist_mean = xy_dist - np.ones(xy_dist.shape[0]) * xy_dist_mean
            xy_dist_mean_max, xy_dist_mean_min = max(dxy_dist_mean), min(dxy_dist_mean)
            xy_dist_sum = sum(xy_dist)

            # Step 4 velocity
            velo = np.abs(xy_dist) / rt
            velo_mean, velo_std = np.mean(velo), np.std(velo)
            max_velo, min_velo = max(velo), min(velo)
            dvelo_mean = velo - np.ones(velo.shape[0]) * velo_mean
            max_velo_mean, min_velo_mean = max(dvelo_mean), min(dvelo_mean)

            x_velo = np.abs(rx) /rt
            x_velo_mean, x_velo_std = np.mean(x_velo), np.std(x_velo)
            max_x_velo, min_x_velo = max(x_velo), min(x_velo)
            dx_velo_mean = x_velo - np.ones(x_velo.shape[0]) * x_velo_mean
            max_x_velo_mean, min_x_velo_mean = max(dx_velo_mean), min(dx_velo_mean)

            y_velo = np.abs(ry) /rt
            y_velo_mean, y_velo_std = np.mean(y_velo), np.std(y_velo)
            max_y_velo, min_y_velo = max(y_velo), min(y_velo)
            dy_velo_mean = y_velo - np.ones(y_velo.shape[0]) * y_velo_mean
            max_y_velo_mean, min_y_velo_mean = max(dy_velo_mean), min(dy_velo_mean)

            if points.shape[0] > 2:
                # Step 5 accelerated velocity
                x_acc_velo, y_acc_velo, xy_acc_velo = np.abs(np.diff(x_velo))/rt[1:], np.abs(np.diff(y_velo))/rt[1:], np.abs(np.diff(velo))/rt[1:]
                x_acc_velo_mean, y_acc_velo_mean, xy_acc_velo_mean = np.mean(x_acc_velo), np.mean(y_acc_velo), np.mean(xy_acc_velo)
                x_acc_velo_std, y_acc_velo_std, xy_acc_velo_std = np.std(x_acc_velo), np.std(y_acc_velo), np.std(xy_acc_velo)
                x_acc_velo_max, y_acc_velo_max, xy_acc_velo_max = max(x_acc_velo), max(y_acc_velo), max(xy_acc_velo)
                x_acc_velo_min, y_acc_velo_min, xy_acc_velo_min = min(x_acc_velo), min(y_acc_velo), min(xy_acc_velo)
                dx_acc_velo = x_acc_velo - np.ones(x_acc_velo.shape[0]) * x_acc_velo_mean
                dy_acc_velo = y_acc_velo - np.ones(y_acc_velo.shape[0]) * y_acc_velo_mean
                dxy_acc_velo = xy_acc_velo - np.ones(xy_acc_velo.shape[0]) * xy_acc_velo_mean
                x_acc_velo_mean_max, y_acc_velo_mean_max, xy_acc_velo_mean_max = max(dx_acc_velo), max(dy_acc_velo), max(dxy_acc_velo)
                x_acc_velo_mean_min, y_acc_velo_mean_min, xy_acc_velo_mean_min = min(dx_acc_velo), min(dy_acc_velo), min(dxy_acc_velo)

                x_dir_acc_velo, y_dir_acc_velo, xy_dir_acc_velo = np.abs(np.diff(x_dir))/rt[1:], np.abs(np.diff(y_dir))/rt[1:], np.abs(np.diff(xy_dir))/rt[1:]
                x_dir_acc_velo_mean, y_dir_acc_velo_mean, xy_dir_acc_velo_mean = np.mean(x_dir_acc_velo), np.mean(y_dir_acc_velo), np.mean(xy_dir_acc_velo)
                x_dir_acc_velo_std, y_dir_acc_velo_std, xy_dir_acc_velo_std = np.std(x_dir_acc_velo), np.std(y_dir_acc_velo), np.std(xy_dir_acc_velo)
                x_dir_acc_velo_max, y_dir_acc_velo_max, xy_dir_acc_velo_max = max(x_dir_acc_velo), max(y_dir_acc_velo), max(xy_dir_acc_velo)
                x_dir_acc_velo_min, y_dir_acc_velo_min, xy_dir_acc_velo_min = min(x_dir_acc_velo), min(y_dir_acc_velo), max(xy_dir_acc_velo)
                dx_dir_acc_velo = x_dir_acc_velo - np.ones(x_dir_acc_velo.shape[0]) * x_dir_acc_velo_mean
                dy_dir_acc_velo = y_dir_acc_velo - np.ones(y_dir_acc_velo.shape[0]) * y_dir_acc_velo_mean
                dxy_dir_acc_velo = xy_dir_acc_velo - np.ones(xy_dir_acc_velo.shape[0]) * xy_dir_acc_velo_mean
                x_dir_acc_velo_mean_max, y_dir_acc_velo_mean_max, xy_dir_acc_velo_mean_max = max(dx_dir_acc_velo), max(dy_dir_acc_velo), max(dxy_dir_acc_velo)
                x_dir_acc_velo_mean_min, y_dir_acc_velo_mean_min, xy_dir_acc_velo_mean_min = min(dx_dir_acc_velo), min(dy_dir_acc_velo), min(dxy_dir_acc_velo)

            else:
                x_acc_velo_mean, x_acc_velo_max, x_acc_velo_min, x_acc_velo_std = -1, -1, -1, -1
                y_acc_velo_mean, y_acc_velo_max, y_acc_velo_min, y_acc_velo_std = -1, -1, -1, -1
                xy_acc_velo_mean, xy_acc_velo_max, xy_acc_velo_min, xy_acc_velo_std = -1, -1, -1, -1
                x_acc_velo_mean_max, y_acc_velo_mean_max, xy_acc_velo_mean_max = -1, -1, -1
                x_acc_velo_mean_min, y_acc_velo_mean_min, xy_acc_velo_mean_min = -1, -1, -1
                x_dir_acc_velo_mean, x_dir_acc_velo_max, x_dir_acc_velo_min, x_dir_acc_velo_std = -1, -1, -1, -1
                y_dir_acc_velo_mean, y_dir_acc_velo_max, y_dir_acc_velo_min, y_dir_acc_velo_std = -1, -1, -1, -1
                xy_dir_acc_velo_mean, xy_dir_acc_velo_max, xy_dir_acc_velo_min, xy_dir_acc_velo_std = -1, -1, -1, -1
                x_dir_acc_velo_mean_max, y_dir_acc_velo_mean_max, xy_dir_acc_velo_mean_max = -1, -1, -1
                x_dir_acc_velo_mean_min, y_dir_acc_velo_mean_min, xy_dir_acc_velo_mean_min = -1, -1, -1
        else:
            x_diff_mean_max, x_diff_mean_min, y_diff_mean_max, y_diff_mean_min, x_std, y_std = -1, -1, -1, -1, -1, -1
            rx_mean, x_resid_max, x_resid_min, x_resid_mean_max, x_resid_mean_min, rx_std = -1, -1, -1, -1, -1, -1
            ry_mean, y_resid_max, y_resid_min, y_resid_mean_max, y_resid_mean_min, ry_std = -1, -1, -1, -1, -1, -1
            rt_mean, t_resid_max, t_resid_min, t_resid_mean_max, t_resid_mean_min, rt_std = -1, -1, -1, -1, -1, -1
            x_dir_mean, x_dir_max, x_dir_min, x_dir_std = -1, -1, -1, -1
            y_dir_mean, y_dir_max, y_dir_min, y_dir_std = -1, -1, -1, -1
            xy_dir_mean, xy_dir_max, xy_dir_min, xy_dir_std = -1, -1, -1, -1
            x_dir_mean_max, y_dir_mean_max, xy_dir_mean_max = -1, -1, -1
            x_dir_mean_min, y_dir_mean_min, xy_dir_mean_min = -1, -1, -1
            xy_dist_mean, xy_dist_max, xy_dist_min, xy_dist_std, xy_dist_sum = -1, -1, -1, -1, -1
            xy_dist_mean_max, xy_dist_mean_min = -1, -1
            velo_mean, max_velo, min_velo, velo_std = -1, -1, -1, -1
            max_velo_mean, min_velo_mean = -1, -1
            x_velo_mean, max_x_velo, min_x_velo, x_velo_std = -1, -1, -1, -1
            max_x_velo_mean, min_x_velo_mean = -1, -1
            y_velo_mean, max_y_velo, min_y_velo, y_velo_std = -1, -1, -1, -1
            max_y_velo_mean, min_y_velo_mean = -1, -1
            x_acc_velo_mean, x_acc_velo_max, x_acc_velo_min, x_acc_velo_std = -1, -1, -1, -1
            y_acc_velo_mean, y_acc_velo_max, y_acc_velo_min, y_acc_velo_std = -1, -1, -1, -1
            xy_acc_velo_mean, xy_acc_velo_max, xy_acc_velo_min, xy_acc_velo_std = -1, -1, -1, -1
            x_dir_acc_velo_mean, x_dir_acc_velo_max, x_dir_acc_velo_min, x_dir_acc_velo_std = -1, -1, -1, -1
            y_dir_acc_velo_mean, y_dir_acc_velo_max, y_dir_acc_velo_min, y_dir_acc_velo_std = -1, -1, -1, -1
            xy_dir_acc_velo_mean, xy_dir_acc_velo_max, xy_dir_acc_velo_min, xy_dir_acc_velo_std = -1, -1, -1, -1

        arr = [k] + \
              [x_std, y_std] + \
              [rx_mean, rx_std] + \
              [ry_mean, ry_std] + \
              [rt_mean, rt_std] + \
              [x_dir_mean, x_dir_std] + \
              [y_dir_mean, y_dir_std] + \
              [xy_dir_mean, xy_dir_std] + \
              [xy_dist_mean, xy_dist_std, xy_dist_sum] + \
              [velo_mean, velo_std] + \
              [x_velo_mean, x_velo_std] + \
              [y_velo_mean, y_velo_std] + \
              [x_acc_velo_mean, x_acc_velo_std] + \
              [y_acc_velo_mean, y_acc_velo_std] + \
              [xy_acc_velo_mean, xy_acc_velo_std] + \
              [x_dir_acc_velo_mean, x_dir_acc_velo_std] + \
              [y_dir_acc_velo_mean, y_dir_acc_velo_std] + \
              [xy_dir_acc_velo_mean, xy_dir_acc_velo_std] + \
              [x_dist, y_dist, p_dist, t_total, points_raw.shape[0]] + \
              [t_before_target, t_after_target, t_per] + \
              [pc_before_target, pc_after_target, pc_per]

        outline = ','.join([str(line) for line in arr[:2]] + [str(round(line, 5)) for line in arr[2:]]) + '\n'

        fw.writelines(outline)

    fw.close()

def main():
    in_file_train, in_file_test = '../data/dsjtzs_txfz_training.txt', '../data/dsjtzs_txfz_test1.txt'
    expand_data = '../data/dsjtzs_txfz_expand_data.txt'
    out_file_train, out_file_test = '../tmp/train_vec_v3.csv', '../tmp/test_vec_v3.csv'
    gen_train(in_file_train=in_file_train, out_file_train=out_file_train)
    gen_train_expand(in_file_train=expand_data, out_file_train=out_file_train)
    # gen_test(in_file_test=in_file_test, out_file_test=out_file_test)

if __name__ == '__main__':
    main()
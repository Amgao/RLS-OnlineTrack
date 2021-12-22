import numpy as np
import multiprocessing
import os
import csv
import math
from itertools import product
from pytracking.evaluation import Sequence, Tracker


def run_sequence(seq: Sequence, tracker: Tracker, iterloop, iou_list, fps_list, result, debug=False, visdom_info=None):
    """Runs a tracker on a sequence."""

    visdom_info = {} if visdom_info is None else visdom_info

    base_results_path = '{}/{}'.format(tracker.results_dir, seq.name)
    if seq.dataset == 'oxuva':
        oxuva_results_path = '{}_{}.csv'.format(base_results_path, seq.object_ids[0])
        results_path = '{}_{}-{}.txt'.format(base_results_path, seq.object_ids[0], str(iterloop))
        times_path = '{}_{}_time-{}.txt'.format(base_results_path, seq.object_ids[0], str(iterloop))
    else:
        results_path = '{}-{}.txt'.format(base_results_path, str(iterloop))
        times_path = '{}_time-{}.txt'.format(base_results_path, str(iterloop))

    #if os.path.isfile(results_path) and not debug:
    #   return

    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))

    if debug:
        output = tracker.run(seq, debug=debug, visdom_info=visdom_info)
    else:
        try:
            if os.path.isfile(results_path):
                output = {'target_bbox': [],
                          'time': []}
                output['target_bbox'] = np.loadtxt(results_path, delimiter='\t', dtype=int)
                output['time'] = np.loadtxt(times_path, delimiter='\t')
            else:
                output = tracker.run(seq, debug=debug, visdom_info=visdom_info)
        except Exception as e:
            print(e)
            return
    if seq.dataset == 'oxuva':
        PREDICTION_FIELD_NAMES = [
            'video', 'object', 'frame_num', 'present', 'score', 'xmin', 'xmax', 'ymin', 'ymax',
        ]
        with open(oxuva_results_path,'w') as f:
            writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELD_NAMES)
            for t, (present, score, xmin, xmax, ymin, ymax) in enumerate(zip(output['present'], output['score'], output['xmin'], output['xmax'], output['ymin'], output['ymax'])):
                row = {
                    'video': seq.name,
                    'object': seq.object_ids[0],
                    'frame_num': seq.start_frame_index + t,
                    'present': present,
                    'score': score,
                    'xmin': xmin,
                    'xmax': xmax,
                    'ymin': ymin,
                    'ymax': ymax,
                }
                writer.writerow(row)
    tracked_bb = np.array(output['target_bbox']).astype(int)
    exec_times = np.array(output['time']).astype(float)
    fps_list[seq.name] = len(exec_times) / exec_times.sum()
    result[seq.name] = tracked_bb
    iou_result = np.zeros((len(seq.frames), 1))

    def overlap_ratio(rect1, rect2):
        '''
        Compute overlap ratio between two rects
        - rect: 1d array of [x,y,w,h] or
                2d array of N x [x,y,w,h]
        '''

        if rect1.ndim == 1:
            rect1 = rect1[None, :]
        if rect2.ndim == 1:
            rect2 = rect2[None, :]

        left = np.maximum(rect1[:, 0], rect2[:, 0])
        right = np.minimum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
        top = np.maximum(rect1[:, 1], rect2[:, 1])
        bottom = np.minimum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])

        intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
        union = rect1[:, 2] * rect1[:, 3] + rect2[:, 2] * rect2[:, 3] - intersect
        iou = np.clip(intersect / union, 0, 1)
        return iou

    iou_result = overlap_ratio(seq.ground_truth_rect, tracked_bb)
    enable_frameNum = 0.
    for iidx in range(len(iou_result)):
        if (math.isnan(iou_result[iidx]) == False):
            enable_frameNum += 1.
        else:
            ## gt is not alowed
            iou_result[iidx] = 0.

    iou_list.append(iou_result.sum() / enable_frameNum)

    # print('FPS: {}'.format(len(exec_times) / exec_times.sum()))
    print('{} {} : {} , total mIoU:{}, fps:{}'.format(len(iou_list), seq.name, iou_result.mean(), sum(iou_list) / len(iou_list),
                                                      sum(fps_list.values()) / len(fps_list)))
    #if not debug:
        #np.savetxt(results_path, tracked_bb, delimiter='\t', fmt='%d')
        #np.savetxt(times_path, exec_times, delimiter='\t', fmt='%f')


def run_dataset(dataset, trackers, debug=False, threads=0, visdom_info=None):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
        visdom_info: Dict containing information about the server for visdom
    """
    print('Evaluating {:4d} trackers on {:5d} sequences'.format(len(trackers), len(dataset)))

    visdom_info = {} if visdom_info is None else visdom_info

    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    if mode == 'sequential':
        mIoU_max = 0.0
        mIoU_min = 1.0
        mIoU_avg = 0.0
        res_list = []
        for iterloop in range(1):
            iou_list = []
            fps_list = dict()
            result = dict()
            for seq in dataset:
                for tracker_info in trackers:
                    run_sequence(seq, tracker_info, iterloop, iou_list, fps_list, result, debug=debug, visdom_info=visdom_info)
            res_list.append(sum(iou_list) / len(iou_list))
            mIoU_avg += sum(iou_list) / len(iou_list)
            if mIoU_max < sum(iou_list) / len(iou_list):
                mIoU_max = sum(iou_list) / len(iou_list)
            if mIoU_min > sum(iou_list) / len(iou_list):
                mIoU_min = sum(iou_list) / len(iou_list)
            np.save(str(iterloop) + 'result_bb', result)
            np.save(str(iterloop) + 'fps', fps_list)
            print(mIoU_max)
            print(mIoU_min)
            print(res_list)
            print(mIoU_avg*1.0/(iterloop+1))
        mIoU_avg /= 1
        print(mIoU_max)
        print(mIoU_avg)
        print(mIoU_min)
        print(res_list)
    elif mode == 'parallel':
        param_list = [(seq, tracker_info, debug, visdom_info) for seq, tracker_info in product(dataset, trackers)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
    print('Done')

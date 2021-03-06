import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList


def UAV10fpsDataset():
    return UAV10fpsDatasetClass().get_sequence_list()


class UAV10fpsDatasetClass(BaseDataset):
    """ UAV123 dataset.

    Publication:
        A Benchmark and Simulator for UAV Tracking.
        Matthias Mueller, Neil Smith and Bernard Ghanem
        ECCV, 2016
        https://ivul.kaust.edu.sa/Documents/Publications/2016/A%20Benchmark%20and%20Simulator%20for%20UAV%20Tracking.pdf

    Download the dataset from https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.uav_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path, 
            sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame+init_omit, end_frame+1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        try:
            ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        except:
            ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)

        return Sequence(sequence_info['name'], frames, ground_truth_rect[init_omit:,:])

    def __len__(self):
        return len(self.sequence_info_list)


    def _get_sequence_info_list(self):
        sequence_info_list = [
            {'name': 'bike1', 'path': 'data_seq/UAV123_10fps/bike1', 'startFrame': 1, 'endFrame': 1029, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/bike1.txt'},
            {'name': 'bike2', 'path': 'data_seq/UAV123_10fps/bike2', 'startFrame': 1, 'endFrame': 185, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/bike2.txt'},
            {'name': 'bike3', 'path': 'data_seq/UAV123_10fps/bike3', 'startFrame': 1, 'endFrame': 145, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/bike3.txt'},
            {'name': 'bird1_1', 'path': 'data_seq/UAV123_10fps/bird1', 'startFrame': 1, 'endFrame': 85, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/bird1_1.txt'},
            {'name': 'bird1_2', 'path': 'data_seq/UAV123_10fps/bird1', 'startFrame': 259, 'endFrame': 493, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/bird1_2.txt'},
            {'name': 'bird1_3', 'path': 'data_seq/UAV123_10fps/bird1', 'startFrame': 525, 'endFrame': 813, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/bird1_3.txt'},
            {'name': 'boat1', 'path': 'data_seq/UAV123_10fps/boat1', 'startFrame': 1, 'endFrame': 301, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/boat1.txt'},
            {'name': 'boat2', 'path': 'data_seq/UAV123_10fps/boat2', 'startFrame': 1, 'endFrame': 267, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/boat2.txt'},
            {'name': 'boat3', 'path': 'data_seq/UAV123_10fps/boat3', 'startFrame': 1, 'endFrame': 301, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/boat3.txt'},
            {'name': 'boat4', 'path': 'data_seq/UAV123_10fps/boat4', 'startFrame': 1, 'endFrame': 185, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/boat4.txt'},
            {'name': 'boat5', 'path': 'data_seq/UAV123_10fps/boat5', 'startFrame': 1, 'endFrame': 169, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/boat5.txt'},
            {'name': 'boat6', 'path': 'data_seq/UAV123_10fps/boat6', 'startFrame': 1, 'endFrame': 269, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/boat6.txt'},
            {'name': 'boat7', 'path': 'data_seq/UAV123_10fps/boat7', 'startFrame': 1, 'endFrame': 179, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/boat7.txt'},
            {'name': 'boat8', 'path': 'data_seq/UAV123_10fps/boat8', 'startFrame': 1, 'endFrame': 229, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/boat8.txt'},
            {'name': 'boat9', 'path': 'data_seq/UAV123_10fps/boat9', 'startFrame': 1, 'endFrame': 467, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/boat9.txt'},
            {'name': 'building1', 'path': 'data_seq/UAV123_10fps/building1', 'startFrame': 1, 'endFrame': 157, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/building1.txt'},
            {'name': 'building2', 'path': 'data_seq/UAV123_10fps/building2', 'startFrame': 1, 'endFrame': 193, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/building2.txt'},
            {'name': 'building3', 'path': 'data_seq/UAV123_10fps/building3', 'startFrame': 1, 'endFrame': 277, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/building3.txt'},
            {'name': 'building4', 'path': 'data_seq/UAV123_10fps/building4', 'startFrame': 1, 'endFrame': 263, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/building4.txt'},
            {'name': 'building5', 'path': 'data_seq/UAV123_10fps/building5', 'startFrame': 1, 'endFrame': 161, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/building5.txt'},
            {'name': 'car1_1', 'path': 'data_seq/UAV123_10fps/car1', 'startFrame': 1, 'endFrame': 251, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/car1_1.txt'},
            {'name': 'car1_2', 'path': 'data_seq/UAV123_10fps/car1', 'startFrame': 251, 'endFrame': 543, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/car1_2.txt'},
            {'name': 'car1_3', 'path': 'data_seq/UAV123_10fps/car1', 'startFrame': 543, 'endFrame': 877, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/car1_3.txt'},
            {'name': 'car2', 'path': 'data_seq/UAV123_10fps/car2', 'startFrame': 1, 'endFrame': 441, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/car2.txt'},
            {'name': 'car3', 'path': 'data_seq/UAV123_10fps/car3', 'startFrame': 1, 'endFrame': 573, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/car3.txt'},
            {'name': 'car4', 'path': 'data_seq/UAV123_10fps/car4', 'startFrame': 1, 'endFrame': 449, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/car4.txt'},
            {'name': 'car5', 'path': 'data_seq/UAV123_10fps/car5', 'startFrame': 1, 'endFrame': 249, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/car5.txt'},
            {'name': 'car6_1', 'path': 'data_seq/UAV123_10fps/car6', 'startFrame': 1, 'endFrame': 163, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/car6_1.txt'},
            {'name': 'car6_2', 'path': 'data_seq/UAV123_10fps/car6', 'startFrame': 163, 'endFrame': 603, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/car6_2.txt'},
            {'name': 'car6_3', 'path': 'data_seq/UAV123_10fps/car6', 'startFrame': 603, 'endFrame': 985, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/car6_3.txt'},
            {'name': 'car6_4', 'path': 'data_seq/UAV123_10fps/car6', 'startFrame': 985, 'endFrame': 1309, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/car6_4.txt'},
            {'name': 'car6_5', 'path': 'data_seq/UAV123_10fps/car6', 'startFrame': 1309, 'endFrame': 1621, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/car6_5.txt'},
            {'name': 'car7', 'path': 'data_seq/UAV123_10fps/car7', 'startFrame': 1, 'endFrame': 345, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/car7.txt'},
            {'name': 'car8_1', 'path': 'data_seq/UAV123_10fps/car8', 'startFrame': 1, 'endFrame': 453, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/car8_1.txt'},
            {'name': 'car8_2', 'path': 'data_seq/UAV123_10fps/car8', 'startFrame': 453, 'endFrame': 859, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/car8_2.txt'},
            {'name': 'car9', 'path': 'data_seq/UAV123_10fps/car9', 'startFrame': 1, 'endFrame': 627, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/car9.txt'},
            {'name': 'car10', 'path': 'data_seq/UAV123_10fps/car10', 'startFrame': 1, 'endFrame': 469, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/car10.txt'},
            {'name': 'car11', 'path': 'data_seq/UAV123_10fps/car11', 'startFrame': 1, 'endFrame': 113, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/car11.txt'},
            {'name': 'car12', 'path': 'data_seq/UAV123_10fps/car12', 'startFrame': 1, 'endFrame': 167, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/car12.txt'},
            {'name': 'car13', 'path': 'data_seq/UAV123_10fps/car13', 'startFrame': 1, 'endFrame': 139, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/car13.txt'},
            {'name': 'car14', 'path': 'data_seq/UAV123_10fps/car14', 'startFrame': 1, 'endFrame': 443, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/car14.txt'},
            {'name': 'car15', 'path': 'data_seq/UAV123_10fps/car15', 'startFrame': 1, 'endFrame': 157, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/car15.txt'},
            {'name': 'car16_1', 'path': 'data_seq/UAV123_10fps/car16', 'startFrame': 1, 'endFrame': 139, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/car16_1.txt'},
            {'name': 'car16_2', 'path': 'data_seq/UAV123_10fps/car16', 'startFrame': 139, 'endFrame': 665, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/car16_2.txt'},
            {'name': 'car17', 'path': 'data_seq/UAV123_10fps/car17', 'startFrame': 1, 'endFrame': 353, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/car17.txt'},
            {'name': 'car18', 'path': 'data_seq/UAV123_10fps/car18', 'startFrame': 1, 'endFrame': 403, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/car18.txt'},
            {'name': 'group1_1', 'path': 'data_seq/UAV123_10fps/group1', 'startFrame': 1, 'endFrame': 445, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/group1_1.txt'},
            {'name': 'group1_2', 'path': 'data_seq/UAV123_10fps/group1', 'startFrame': 445, 'endFrame': 839, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/group1_2.txt'},
            {'name': 'group1_3', 'path': 'data_seq/UAV123_10fps/group1', 'startFrame': 839, 'endFrame': 1309, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/group1_3.txt'},
            {'name': 'group1_4', 'path': 'data_seq/UAV123_10fps/group1', 'startFrame': 1309, 'endFrame': 1625, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/group1_4.txt'},
            {'name': 'group2_1', 'path': 'data_seq/UAV123_10fps/group2', 'startFrame': 1, 'endFrame': 303, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/group2_1.txt'},
            {'name': 'group2_2', 'path': 'data_seq/UAV123_10fps/group2', 'startFrame': 303, 'endFrame': 591, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/group2_2.txt'},
            {'name': 'group2_3', 'path': 'data_seq/UAV123_10fps/group2', 'startFrame': 591, 'endFrame': 895, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/group2_3.txt'},
            {'name': 'group3_1', 'path': 'data_seq/UAV123_10fps/group3', 'startFrame': 1, 'endFrame': 523, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/group3_1.txt'},
            {'name': 'group3_2', 'path': 'data_seq/UAV123_10fps/group3', 'startFrame': 523, 'endFrame': 943, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/group3_2.txt'},
            {'name': 'group3_3', 'path': 'data_seq/UAV123_10fps/group3', 'startFrame': 943, 'endFrame': 1457, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/group3_3.txt'},
            {'name': 'group3_4', 'path': 'data_seq/UAV123_10fps/group3', 'startFrame': 1457, 'endFrame': 1843, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/group3_4.txt'},
            {'name': 'person1', 'path': 'data_seq/UAV123_10fps/person1', 'startFrame': 1, 'endFrame': 267, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person1.txt'},
            {'name': 'person2_1', 'path': 'data_seq/UAV123_10fps/person2', 'startFrame': 1, 'endFrame': 397, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person2_1.txt'},
            {'name': 'person2_2', 'path': 'data_seq/UAV123_10fps/person2', 'startFrame': 397, 'endFrame': 875, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person2_2.txt'},
            {'name': 'person3', 'path': 'data_seq/UAV123_10fps/person3', 'startFrame': 1, 'endFrame': 215, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person3.txt'},
            {'name': 'person4_1', 'path': 'data_seq/UAV123_10fps/person4', 'startFrame': 1, 'endFrame': 501, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person4_1.txt'},
            {'name': 'person4_2', 'path': 'data_seq/UAV123_10fps/person4', 'startFrame': 501, 'endFrame': 915, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person4_2.txt'},
            {'name': 'person5_1', 'path': 'data_seq/UAV123_10fps/person5', 'startFrame': 1, 'endFrame': 293, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person5_1.txt'},
            {'name': 'person5_2', 'path': 'data_seq/UAV123_10fps/person5', 'startFrame': 293, 'endFrame': 701, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person5_2.txt'},
            {'name': 'person6', 'path': 'data_seq/UAV123_10fps/person6', 'startFrame': 1, 'endFrame': 301, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person6.txt'},
            {'name': 'person7_1', 'path': 'data_seq/UAV123_10fps/person7', 'startFrame': 1, 'endFrame': 417, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person7_1.txt'},
            {'name': 'person7_2', 'path': 'data_seq/UAV123_10fps/person7', 'startFrame': 417, 'endFrame': 689, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person7_2.txt'},
            {'name': 'person8_1', 'path': 'data_seq/UAV123_10fps/person8', 'startFrame': 1, 'endFrame': 359, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person8_1.txt'},
            {'name': 'person8_2', 'path': 'data_seq/UAV123_10fps/person8', 'startFrame': 359, 'endFrame': 509, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person8_2.txt'},
            {'name': 'person9', 'path': 'data_seq/UAV123_10fps/person9', 'startFrame': 1, 'endFrame': 221, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person9.txt'},
            {'name': 'person10', 'path': 'data_seq/UAV123_10fps/person10', 'startFrame': 1, 'endFrame': 341, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person10.txt'},
            {'name': 'person11', 'path': 'data_seq/UAV123_10fps/person11', 'startFrame': 1, 'endFrame': 241, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person11.txt'},
            {'name': 'person12_1', 'path': 'data_seq/UAV123_10fps/person12', 'startFrame': 1, 'endFrame': 201, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person12_1.txt'},
            {'name': 'person12_2', 'path': 'data_seq/UAV123_10fps/person12', 'startFrame': 201, 'endFrame': 541,
             'nz': 6, 'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person12_2.txt'},
            {'name': 'person13', 'path': 'data_seq/UAV123_10fps/person13', 'startFrame': 1, 'endFrame': 295, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person13.txt'},
            {'name': 'person14_1', 'path': 'data_seq/UAV123_10fps/person14', 'startFrame': 1, 'endFrame': 283, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person14_1.txt'},
            {'name': 'person14_2', 'path': 'data_seq/UAV123_10fps/person14', 'startFrame': 283, 'endFrame': 605,
             'nz': 6, 'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person14_2.txt'},
            {'name': 'person14_3', 'path': 'data_seq/UAV123_10fps/person14', 'startFrame': 605, 'endFrame': 975,
             'nz': 6, 'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person14_3.txt'},
            {'name': 'person15', 'path': 'data_seq/UAV123_10fps/person15', 'startFrame': 1, 'endFrame': 447, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person15.txt'},
            {'name': 'person16', 'path': 'data_seq/UAV123_10fps/person16', 'startFrame': 1, 'endFrame': 383, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person16.txt'},
            {'name': 'person17_1', 'path': 'data_seq/UAV123_10fps/person17', 'startFrame': 1, 'endFrame': 501, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person17_1.txt'},
            {'name': 'person17_2', 'path': 'data_seq/UAV123_10fps/person17', 'startFrame': 501, 'endFrame': 783,
             'nz': 6, 'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person17_2.txt'},
            {'name': 'person18', 'path': 'data_seq/UAV123_10fps/person18', 'startFrame': 1, 'endFrame': 465, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person18.txt'},
            {'name': 'person19_1', 'path': 'data_seq/UAV123_10fps/person19', 'startFrame': 1, 'endFrame': 415, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person19_1.txt'},
            {'name': 'person19_2', 'path': 'data_seq/UAV123_10fps/person19', 'startFrame': 415, 'endFrame': 931,
             'nz': 6, 'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person19_2.txt'},
            {'name': 'person19_3', 'path': 'data_seq/UAV123_10fps/person19', 'startFrame': 931, 'endFrame': 1453,
             'nz': 6, 'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person19_3.txt'},
            {'name': 'person20', 'path': 'data_seq/UAV123_10fps/person20', 'startFrame': 1, 'endFrame': 595, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person20.txt'},
            {'name': 'person21', 'path': 'data_seq/UAV123_10fps/person21', 'startFrame': 1, 'endFrame': 163, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person21.txt'},
            {'name': 'person22', 'path': 'data_seq/UAV123_10fps/person22', 'startFrame': 1, 'endFrame': 67, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person22.txt'},
            {'name': 'person23', 'path': 'data_seq/UAV123_10fps/person23', 'startFrame': 1, 'endFrame': 133, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person23.txt'},
            {'name': 'truck1', 'path': 'data_seq/UAV123_10fps/truck1', 'startFrame': 1, 'endFrame': 155, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/truck1.txt'},
            {'name': 'truck2', 'path': 'data_seq/UAV123_10fps/truck2', 'startFrame': 1, 'endFrame': 129, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/truck2.txt'},
            {'name': 'truck3', 'path': 'data_seq/UAV123_10fps/truck3', 'startFrame': 1, 'endFrame': 179, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/truck3.txt'},
            {'name': 'truck4_1', 'path': 'data_seq/UAV123_10fps/truck4', 'startFrame': 1, 'endFrame': 193, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/truck4_1.txt'},
            {'name': 'truck4_2', 'path': 'data_seq/UAV123_10fps/truck4', 'startFrame': 193, 'endFrame': 421, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/truck4_2.txt'},
            {'name': 'uav1_1', 'path': 'data_seq/UAV123_10fps/uav1', 'startFrame': 1, 'endFrame': 519, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/uav1_1.txt'},
            {'name': 'uav1_2', 'path': 'data_seq/UAV123_10fps/uav1', 'startFrame': 519, 'endFrame': 793, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/uav1_2.txt'},
            {'name': 'uav1_3', 'path': 'data_seq/UAV123_10fps/uav1', 'startFrame': 825, 'endFrame': 1157, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/uav1_3.txt'},
            {'name': 'uav2', 'path': 'data_seq/UAV123_10fps/uav2', 'startFrame': 1, 'endFrame': 45, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/uav2.txt'},
            {'name': 'uav3', 'path': 'data_seq/UAV123_10fps/uav3', 'startFrame': 1, 'endFrame': 89, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/uav3.txt'},
            {'name': 'uav4', 'path': 'data_seq/UAV123_10fps/uav4', 'startFrame': 1, 'endFrame': 53, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/uav4.txt'},
            {'name': 'uav5', 'path': 'data_seq/UAV123_10fps/uav5', 'startFrame': 1, 'endFrame': 47, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/uav5.txt'},
            {'name': 'uav6', 'path': 'data_seq/UAV123_10fps/uav6', 'startFrame': 1, 'endFrame': 37, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/uav6.txt'},
            {'name': 'uav7', 'path': 'data_seq/UAV123_10fps/uav7', 'startFrame': 1, 'endFrame': 125, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/uav7.txt'},
            {'name': 'uav8', 'path': 'data_seq/UAV123_10fps/uav8', 'startFrame': 1, 'endFrame': 101, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/uav8.txt'},
            {'name': 'wakeboard1', 'path': 'data_seq/UAV123_10fps/wakeboard1', 'startFrame': 1, 'endFrame': 141,
             'nz': 6, 'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/wakeboard1.txt'},
            {'name': 'wakeboard2', 'path': 'data_seq/UAV123_10fps/wakeboard2', 'startFrame': 1, 'endFrame': 245,
             'nz': 6, 'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/wakeboard2.txt'},
            {'name': 'wakeboard3', 'path': 'data_seq/UAV123_10fps/wakeboard3', 'startFrame': 1, 'endFrame': 275,
             'nz': 6, 'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/wakeboard3.txt'},
            {'name': 'wakeboard4', 'path': 'data_seq/UAV123_10fps/wakeboard4', 'startFrame': 1, 'endFrame': 233,
             'nz': 6, 'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/wakeboard4.txt'},
            {'name': 'wakeboard5', 'path': 'data_seq/UAV123_10fps/wakeboard5', 'startFrame': 1, 'endFrame': 559,
             'nz': 6, 'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/wakeboard5.txt'},
            {'name': 'wakeboard6', 'path': 'data_seq/UAV123_10fps/wakeboard6', 'startFrame': 1, 'endFrame': 389,
             'nz': 6, 'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/wakeboard6.txt'},
            {'name': 'wakeboard7', 'path': 'data_seq/UAV123_10fps/wakeboard7', 'startFrame': 1, 'endFrame': 67, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/wakeboard7.txt'},
            {'name': 'wakeboard8', 'path': 'data_seq/UAV123_10fps/wakeboard8', 'startFrame': 1, 'endFrame': 515,
             'nz': 6, 'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/wakeboard8.txt'},
            {'name': 'wakeboard9', 'path': 'data_seq/UAV123_10fps/wakeboard9', 'startFrame': 1, 'endFrame': 119,
             'nz': 6, 'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/wakeboard9.txt'},
            {'name': 'wakeboard10', 'path': 'data_seq/UAV123_10fps/wakeboard10', 'startFrame': 1, 'endFrame': 157,
             'nz': 6, 'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/wakeboard10.txt'},
            {'name': 'car1_s', 'path': 'data_seq/UAV123_10fps/car1_s', 'startFrame': 1, 'endFrame': 492, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/car1_s.txt'},
            {'name': 'car2_s', 'path': 'data_seq/UAV123_10fps/car2_s', 'startFrame': 1, 'endFrame': 107, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/car2_s.txt'},
            {'name': 'car3_s', 'path': 'data_seq/UAV123_10fps/car3_s', 'startFrame': 1, 'endFrame': 434, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/car3_s.txt'},
            {'name': 'car4_s', 'path': 'data_seq/UAV123_10fps/car4_s', 'startFrame': 1, 'endFrame': 277, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/car4_s.txt'},
            {'name': 'person1_s', 'path': 'data_seq/UAV123_10fps/person1_s', 'startFrame': 1, 'endFrame': 534, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person1_s.txt'},
            {'name': 'person2_s', 'path': 'data_seq/UAV123_10fps/person2_s', 'startFrame': 1, 'endFrame': 84, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person2_s.txt'},
            {'name': 'person3_s', 'path': 'data_seq/UAV123_10fps/person3_s', 'startFrame': 1, 'endFrame': 169, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV123_10fps/person3_s.txt'}
        ]

        return sequence_info_list

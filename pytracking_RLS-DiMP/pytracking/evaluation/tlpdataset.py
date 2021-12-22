import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
from pytracking.utils.load_text import load_text

def TLPDataset():
    return TLPDatasetClass().get_sequence_list()

class TLPDatasetClass(BaseDataset):
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.tlp_path
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

        # NOTE: OTB has some weird annos which panda cannot handle
        ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')

        return Sequence(sequence_info['name'], frames, ground_truth_rect[init_omit:,1:5])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {'name': 'MotorcycleChase', 'path': 'MotorcycleChase/img', 'anno_path': 'MotorcycleChase/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 5550, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Boxing3', 'path': 'Boxing3/img', 'anno_path': 'Boxing3/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 19590, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'BreakfastClub', 'path': 'BreakfastClub/img', 'anno_path': 'BreakfastClub/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 22600, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Hideaway', 'path': 'Hideaway/img', 'anno_path': 'Hideaway/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 5900, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Boxing2', 'path': 'Boxing2/img', 'anno_path': 'Boxing2/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 21180, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Mohiniyattam', 'path': 'Mohiniyattam/img', 'anno_path': 'Mohiniyattam/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 15456, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Drone2', 'path': 'Drone2/img', 'anno_path': 'Drone2/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 8812, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Alladin', 'path': 'Alladin/img', 'anno_path': 'Alladin/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 8992, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Puppies1', 'path': 'Puppies1/img', 'anno_path': 'Puppies1/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 17730, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Badminton2', 'path': 'Badminton2/img', 'anno_path': 'Badminton2/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 16920, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'ZebraFish', 'path': 'ZebraFish/img', 'anno_path': 'ZebraFish/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 10920, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'PolarBear2', 'path': 'PolarBear2/img', 'anno_path': 'PolarBear2/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 27153, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'KinBall1', 'path': 'KinBall1/img', 'anno_path': 'KinBall1/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 20230, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Dashcam', 'path': 'Dashcam/img', 'anno_path': 'Dashcam/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 10260, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Jet4', 'path': 'Jet4/img', 'anno_path': 'Jet4/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 10160, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'IceSkating', 'path': 'IceSkating/img', 'anno_path': 'IceSkating/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 8125, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Rope', 'path': 'Rope/img', 'anno_path': 'Rope/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 17503, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'ISS', 'path': 'ISS/img', 'anno_path': 'ISS/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 28562, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Elephants', 'path': 'Elephants/img', 'anno_path': 'Elephants/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 4376, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'CarChase1', 'path': 'CarChase1/img', 'anno_path': 'CarChase1/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 8932, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Jet2', 'path': 'Jet2/img', 'anno_path': 'Jet2/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 18882, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'KinBall3', 'path': 'KinBall3/img', 'anno_path': 'KinBall3/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 14940, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Drone1', 'path': 'Drone1/img', 'anno_path': 'Drone1/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 4320, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Aquarium1', 'path': 'Aquarium1/img', 'anno_path': 'Aquarium1/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 7337, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Jet1', 'path': 'Jet1/img', 'anno_path': 'Jet1/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 7403, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Violinist', 'path': 'Violinist/img', 'anno_path': 'Violinist/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 6844, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'KinBall2', 'path': 'KinBall2/img', 'anno_path': 'KinBall2/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 13575, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Billiards2', 'path': 'Billiards2/img', 'anno_path': 'Billiards2/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 20070, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Puppies2', 'path': 'Puppies2/img', 'anno_path': 'Puppies2/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 22620, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'DriftCar2', 'path': 'DriftCar2/img', 'anno_path': 'DriftCar2/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 8572, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Lion', 'path': 'Lion/img', 'anno_path': 'Lion/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 6570, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Billiards1', 'path': 'Billiards1/img', 'anno_path': 'Billiards1/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 20375, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Jet3', 'path': 'Jet3/img', 'anno_path': 'Jet3/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 17953, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Bike', 'path': 'Bike/img', 'anno_path': 'Bike/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 4196, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Helicopter', 'path': 'Helicopter/img', 'anno_path': 'Helicopter/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 17053, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Parakeet', 'path': 'Parakeet/img', 'anno_path': 'Parakeet/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 21609, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Badminton1', 'path': 'Badminton1/img', 'anno_path': 'Badminton1/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 15240, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'PolarBear1', 'path': 'PolarBear1/img', 'anno_path': 'PolarBear1/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 9501, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'CarChase2', 'path': 'CarChase2/img', 'anno_path': 'CarChase2/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 14010, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Basketball', 'path': 'Basketball/img', 'anno_path': 'Basketball/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 17970, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Aquarium2', 'path': 'Aquarium2/img', 'anno_path': 'Aquarium2/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 8182, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Boxing1', 'path': 'Boxing1/img', 'anno_path': 'Boxing1/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 20670, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Sam', 'path': 'Sam/img', 'anno_path': 'Sam/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 4628, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'CarChase3', 'path': 'CarChase3/img', 'anno_path': 'CarChase3/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 22860, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Boat', 'path': 'Boat/img', 'anno_path': 'Boat/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 6234, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'DriftCar1', 'path': 'DriftCar1/img', 'anno_path': 'DriftCar1/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 10130, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Drone3', 'path': 'Drone3/img', 'anno_path': 'Drone3/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 6594, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Jet5', 'path': 'Jet5/img', 'anno_path': 'Jet5/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 13675, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'PolarBear3', 'path': 'PolarBear3/img', 'anno_path': 'PolarBear3/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 9531, 'nz': 5, 'ext': 'jpg'} ,
            {'name': 'Bharatanatyam', 'path': 'Bharatanatyam/img', 'anno_path': 'Bharatanatyam/groundtruth_rect.txt', 'startFrame': 1, 'endFrame': 15936, 'nz': 5, 'ext': 'jpg'} ,

        ]
    
        return sequence_info_list

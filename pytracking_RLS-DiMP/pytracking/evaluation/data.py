from pytracking.evaluation.environment import env_settings


class BaseDataset:
    """Base class for all datasets."""
    def __init__(self):
        self.env_settings = env_settings()

    def __len__(self):
        """Overload this function in your dataset. This should return number of sequences in the dataset."""
        raise NotImplementedError

    def get_sequence_list(self):
        """Overload this in your dataset. Should return the list of sequences in the dataset."""
        raise NotImplementedError


class Sequence:
    """Class for the sequence in an evaluation."""
    def __init__(self, name, frames, ground_truth_rect, dataset=None, object_ids=None):
        self.name = name
        self.dataset = dataset
        self.frames = frames
        self.ground_truth_rect = ground_truth_rect
        self.object_ids = object_ids
        self.start_frame_index = int(self.frames[0].split('.')[0].split('/')[-1])

    def init_info(self):
        return {key: self.get(key) for key in ['init_bbox']}

    def init_bbox(self):
        return list(self.ground_truth_rect[0,:])

    def get(self, name):
        return getattr(self, name)()



class SequenceList(list):
    """List of sequences. Supports the addition operator to concatenate sequence lists."""
    def __getitem__(self, item):
        if isinstance(item, str):
            for seq in self:
                if seq.name == item:
                    return seq
            raise IndexError('Sequence name not in the dataset.')
        elif isinstance(item, int):
            return super(SequenceList, self).__getitem__(item)
        elif isinstance(item, (tuple, list)):
            return SequenceList([super(SequenceList, self).__getitem__(i) for i in item])
        else:
            return SequenceList(super(SequenceList, self).__getitem__(item))

    def __add__(self, other):
        return SequenceList(super(SequenceList, self).__add__(other))

    def copy(self):
        return SequenceList(super(SequenceList, self).copy())
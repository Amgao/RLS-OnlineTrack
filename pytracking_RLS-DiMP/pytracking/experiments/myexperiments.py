from pytracking.evaluation import Tracker, OTBDataset, NFSDataset, UAVDataset, TPLDataset, VOTDataset, TrackingNetDataset, LaSOTDataset, OxuvaDataset, TLPDataset


def atom_nfs_uav():
    # Run three runs of ATOM on NFS and UAV datasets
    trackers = [Tracker('atom', 'default', i) for i in range(3)]

    dataset = NFSDataset() + UAVDataset()
    return trackers, dataset


def uav_test():
    # Run ATOM and ECO on the UAV dataset
    #trackers = [Tracker('atom', 'default', i) for i in range(1)] + \
    #           [Tracker('eco', 'default', i) for i in range(1)]
    trackers = [Tracker('atom', 'default', i) for i in range(1)]
    dataset = UAVDataset()
    return trackers, dataset


def otb_test():
    # Run ATOM and ECO on the UAV dataset
    trackers = [Tracker('atom', 'default', i) for i in range(1)]

    dataset = OTBDataset()
    return trackers, dataset

def lasot_test():
    # Run ATOM and ECO on the UAV dataset
    trackers = [Tracker('atom', 'default', i) for i in range(1)]

    dataset = LaSOTDataset()
    return trackers, dataset

def trackingnet_test():
    # Run ATOM and ECO on the UAV dataset
    trackers = [Tracker('dimp', 'dimp50', i) for i in range(1)]

    dataset = TrackingNetDataset()
    return trackers, dataset

def dimp_otb_test():
    # Run ATOM and ECO on the UAV dataset
    trackers = [Tracker('dimp', 'dimp50', i) for i in range(50)]

    dataset = OTBDataset()
    return trackers, dataset

def dimp_uav_test():
    # Run ATOM and ECO on the UAV dataset
    #trackers = [Tracker('atom', 'default', i) for i in range(1)] + \
    #           [Tracker('eco', 'default', i) for i in range(1)]
    trackers = [Tracker('dimp', 'dimp18', i) for i in range(1)]
    dataset = UAVDataset()
    return trackers, dataset

def dimp50_uav_test():
    # Run ATOM and ECO on the UAV dataset
    #trackers = [Tracker('atom', 'default', i) for i in range(1)] + \
    #           [Tracker('eco', 'default', i) for i in range(1)]
    trackers = [Tracker('dimp', 'dimp50', i) for i in range(50)]
    dataset = UAVDataset()
    return trackers, dataset

def dimp50_lasot_test():
    # Run ATOM and ECO on the UAV dataset
    trackers = [Tracker('dimp', 'dimp50', i) for i in range(20)]

    dataset = LaSOTDataset()
    return trackers, dataset

def dimp50_oxuva_test():
    # Run ATOM and ECO on the UAV dataset
    trackers = [Tracker('dimp', 'dimp50_oxuva', i) for i in range(1)]

    dataset = OxuvaDataset()
    return trackers, dataset

def dimp50_tlp_test():
    # Run ATOM and ECO on the UAV dataset
    trackers = [Tracker('dimp', 'dimp50', i) for i in range(20)]

    dataset = TLPDataset()
    return trackers, dataset

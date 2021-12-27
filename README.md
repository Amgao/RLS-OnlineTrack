# RLS-OnlineTrack
Code and raw result files of our paper "Recursive Least-Squares Estimator-Aided Online Learning for Visual Tracking"

Created by [Jin Gao](http://people.ucas.ac.cn/~jgao?language=en)

### Introduction
RLS-OnlineTrack is dedicated to improving online tracking parts of both RT-MDNet ([project page](https://github.com/IlchaeJung/RT-MDNet) and [paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ilchae_Jung_Real-Time_MDNet_ECCV_2018_paper.pdf)) and DiMP ([project page](https://github.com/visionml/pytracking) and [paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Bhat_Learning_Discriminative_Model_Prediction_for_Tracking_ICCV_2019_paper.pdf)) based on our proposed recursive least-squares estimator-aided online learning method.

### Citation
If you're using this code in a publication, please cite [our paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Gao_Recursive_Least-Squares_Estimator-Aided_Online_Learning_for_Visual_Tracking_CVPR_2020_paper.html).

	@InProceedings{Gao_2020_CVPR,
   	author = {Gao, Jin and Hu, Weiming and Lu, Yan},
    	title = {Recursive Least-squares Estimator-aided Online Learning for Visual Tracking},
    	booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    	month = {June},
    	year = {2020}
  	}
  

### System Requirements

The above codes are tested on 64 bit Linux (Ubuntu 18.04 LTS) with the following Anaconda environment:
>> * PyTorch (= 1.7.1+cu110)
>> * Python (= 3.7.4)

### Acknowledgement

The above codes are modified based on the following repositories::
>> * [RT-MDNet](https://github.com/IlchaeJung/RT-MDNet)
>> * [pytracking](https://github.com/visionml/pytracking)

### Online Tracking

**Pretrained Model**
>> * The off-the-shelf pretrained model in RT-MDNet is used for our testing: [RT-MDNet-ImageNet-pretrained](https://www.dropbox.com/s/lr8uft05zlo21an/rt-mdnet.pth?dl=0).
>> * The off-the-shelf pretrained model in DiMP is used for our testing: [dimp50.pth](https://github.com/visionml/pytracking/blob/master/MODEL_ZOO.md#Models).
>> * The pretrained DiMP model by ourselves following the GOT10k protocol: [dimp50_got10k.pth.tar](https://drive.google.com/file/d/1KlE2SU6qoCtV2Md4ueX51ACnSJ7zCNUJ/view?usp=sharing).

**Demo for RLS-RTMDNet**
>> * 'Run.py' for OTB and UAV123
>> * 'python_RLS_RTMDNet.py' for VOT16/17.

**Demo for RLS-DiMP**
>> * 'run_experiment.py myexperiments dimp50_XX_test' for TrackingNet, GOT10k, LaSOT, OxUvA, and TLP

# DukeMTMC-VideoReID

DukeMTMC-VideoReID is a subset of the [DukeMTMC](http://vision.cs.duke.edu/DukeMTMC/) for video-based re-identification.

We crop pedestrian images from the videos for 12 frames every second to generate a tracklet. The dataset is split following 
the protocol in [DukeMTMC-reID](https://github.com/layumi/DukeMTMC-reID_evaluation), i.e., 702 identities for training, 702 
identities for testing, and 408 identities as the distractors. Totally, we generate 369,656 frames of 2,196 tracklets for 
training, and 445,764 frames of 2,636 tracklets for testing and distractors.
In the testing set, we pick one query tracklet for each ID in each camera and put the remaining tracklets in the gallery. 


### About Dataset
|File  | Description | 
| --------   | -----  |
|./train_split  | The training tracklets. It contains 702 identities.|
|./query_split  | The query tracklets. Each of them is from different identities in different cameras.|
|./gallery_split  | The gallery_split tracklets.|

**Naming Rules.** For frame "0001C6F0099X30823.jpg" from tracklet id 0003 of person id 0001,
"0001" is the identity. "C6" indicate it from Camera 6. "F0099" means it is the 99th frame in the tracklet. 
"X30823" is the 30823th frame in the video of Camera 6.

### Download Dataset

You can download the DukeMTMC-VideoReID dataset from 
[[Google Drive]](https://drive.google.com/open?id=1JBrffnNTZufQ-hRYr9I8FtxeGqDkud1r), 
[[BaiduYun]](https://pan.baidu.com/),
or [[Direct Link]](http://45.62.126.139:8080/dukemtmc_videoReID.zip).


### Baseline
We will release the baseline code soon.

### References
```
@inproceedings{wu2018cvpr_oneshot,
  title = {Exploit the Unknown Gradually: One-Shot Video-Based Person Re-Identification by Stepwise Learning},
  author = {Wu, Yu and Lin, Yutian and Dong, Xuanyi and Yan, Yan and Ouyang, Wanli and Yang, Yi},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2018}
}
```
If you use this dataset, please also cite the original DukeMTMC dataset accordingly:
```
@inproceedings{ristani2016MTMC,
  title = {Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking},
  author = {Ristani, Ergys and Solera, Francesco and Zou, Roger and Cucchiara, Rita and Tomasi, Carlo},
  booktitle = {European Conference on Computer Vision workshop on Benchmarking Multi-Target Tracking},
  year = {2016}
}
```
### License
Please refer to the license file for [DukeMTMC-VideoReID](https://github.com/Yu-Wu/DukeMTMC-VideoReID/blob/master/LICENSE_DukeMTMC-VideoReID.txt) and [DukeMTMC](https://github.com/Yu-Wu/DukeMTMC-VideoReID/blob/master/LICENSE_DukeMTMC.txt).

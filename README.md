
# DukeMTMC-VideoReID

DukeMTMC-VideoReID [1] is a subset of the [DukeMTMC](http://vision.cs.duke.edu/DukeMTMC/) tracking dataset [2] for video-based person re-identification.


The dataset consists of 702 identities for training, 702 identities for testing, and 408 identities as distractors. In total there are 2,196 videos for training and 2,636 videos for testing. Each video contains person images sampled every 12 frames. During testing, a video for each ID is used as the query and the remaining videos are placed in the gallery.


### Download Dataset
You can download the DukeMTMC-VideoReID dataset from
[[Direct Link]](http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-VideoReID.zip)  [[Google Drive]](https://drive.google.com/open?id=1WVjJ7PwhakF40a-BgOs1Jr_a17O38eOz)  [[BaiduYun]](https://pan.baidu.com/s/1Y_unlSqZqSdU3SeBqQmE5A)   [[Direct Link 2]](http://45.62.126.139/DukeMTMC-VideoReID.zip).


### About Dataset
|Directory  | Description | 
| --------   | -----  |
|./train  | The training video tracklets. It contains 702 identities.|
|./query  | The query video tracklets. Each of them is from different identities in different cameras.|
|./gallery  | The gallery video tracklets. It contains 702 gallery identities and 408 distractors.|

### Directory Structure
Followings are the directory structure for DukeMTMC-VideoReID. 
> Splits
>> Person ids
>>> Video tracklet ids
>>>> Frame bounding box images

For example, for one frame image `train/0001/0003/0001C6F0099X30823.jpg`, `train`, `0001`, `0003`, and `0001C6F0099X30823.jpg` are the split, person id, video tracklet id, and image frame name, respectively.

**Naming Rules for image file.** 
For the frame bounding box image `0001C6F0099X30823.jpg`, "0001" is the identity. "C6" indicate Camera 6. "F0099" means it is the 99th frame within the tracklet. "X30823" is the 30823th frame in the whole video of Camera 6.


## Training Baseline model

### ETAP-Net
The baseline model is an end-to-end ResNet-50 model with temporal average pooling (ETAP-Net).

More details about the ETAP-Net can be found in [Exploit the Unknown Gradually: One-Shot Video-Based Person Re-Identification by Stepwise Learning](https://yu-wu.net/pdf/CVPR2018_Exploit-Unknown-Gradually.pdf).


### Dependencies
- Python 3.6
- PyTorch (version >= 0.2.0)
- h5py, scikit-learn, metric-learn, tqdm

### Run
Move the downloaded dataset file `DukeMTMC-VideoReID.zip` to `./data/` and unzip here.

```shell
python3 run.py --dataset DukeMTMC-VideoReID --logs_dir logs/DukeMTMC-VideoReID_baseline/ --max_frames 900 --batch_size 16
```

### Results on DukeMTMC-VideoReID

<table>
   <tr>
      <td>Method</td>
      <td>Rank-1</td>
      <td>Rank-5</td>
      <td>Rank-20</td>
      <td>mAP</td>
   </tr>
   <tr>
      <td>ETAP-Net</td>
      <td>83.62 </td>
      <td>94.59</td>
      <td>97.58</td>
      <td>78.34</td>
   </tr>
</table>

### References
- [1] Exploit the Unknown Gradually: One-Shot Video-Based Person Re-Identification by Stepwise Learning. Wu et al., CVPR 2018
 
- [2] Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking. Ristani et al., ECCVWS 2016

Please cite the following two papers if this dataset helps your research.
```
@inproceedings{wu2018cvpr_oneshot,
  title = {Exploit the Unknown Gradually: One-Shot Video-Based Person Re-Identification by Stepwise Learning},
  author = {Wu, Yu and Lin, Yutian and Dong, Xuanyi and Yan, Yan and Ouyang, Wanli and Yang, Yi},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2018}
}

@inproceedings{ristani2016MTMC,
  title = {Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking},
  author = {Ristani, Ergys and Solera, Francesco and Zou, Roger and Cucchiara, Rita and Tomasi, Carlo},
  booktitle = {European Conference on Computer Vision workshop on Benchmarking Multi-Target Tracking},
  year = {2016}
}
```
### License
Please refer to the license file for [DukeMTMC-VideoReID](https://github.com/Yu-Wu/DukeMTMC-VideoReID/blob/master/LICENSES/LICENSE_DukeMTMC-VideoReID.txt) and [DukeMTMC](https://github.com/Yu-Wu/DukeMTMC-VideoReID/blob/master/LICENSES/LICENSE_DukeMTMC.txt).


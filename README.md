
# DukeMTMC-VideoReID

DukeMTMC-VideoReID is a subset of the [DukeMTMC](http://vision.cs.duke.edu/DukeMTMC/) for video-based person re-identification.

We crop pedestrian images from the videos for 12 frames every second to generate a tracklet. The dataset is split following 
the protocol in [DukeMTMC-reID](https://github.com/layumi/DukeMTMC-reID_evaluation), i.e., 702 identities for training, 702 
identities for testing, and 408 identities as the distractors. Totally, we generate 369,656 frames of 2,196 tracklets for 
training, and 445,764 frames of 2,636 tracklets for testing and distractors.
In the testing set, we pick one query tracklet for each ID in each camera and put the remaining tracklets in the gallery. 


### Download Dataset
You can download the DukeMTMC-VideoReID dataset from
[[Google Drive]](https://drive.google.com/open?id=1T5bmWetLSvLjR30hAp8S54G2ERs81Pkg) [[BaiduYun]](https://pan.baidu.com/s/1axx55Z7XDzc95i0yGr4ItQ) [[Direct Link]](http://45.62.126.139/dukemtmc_videoReID.zip).


### About Dataset
|File  | Description | 
| --------   | -----  |
|./train_split  | The training tracklets. It contains 702 identities.|
|./query_split  | The query tracklets. Each of them is from different identities in different cameras.|
|./gallery_split  | The gallery_split tracklets. It contains 702 gallery identities and 408 distractors.|

### Directory Structure
Followings are the directory structure for DukeMTMC-VideoReID. 
> Splits
>> Person ids
>>> Tracklet ids
>>>> Frame bounding box images

For example, for one frame image `train_split/0001/0003/0001C6F0099X30823.jpg`, `train_split`, `0001`, `0003`, and `0001C6F0099X30823.jpg` are the split, person id, tracklet id, and image frame name, respectively.

**Naming Rules for image file.** 
For the frame bounding box image `0001C6F0099X30823.jpg`, "0001" is the identity. "C6" indicate Camera 6. "F0099" means it is the 99th frame within the tracklet. "X30823" is the 30823th frame in the whole video of Camera 6.


## Training Baseline model

### ETAP-Net
The baseline model is an end-to-end ResNet-50 model with temporal average pooling (ETAP-Net).

More details about the ETAP-Net is illustrated in [Exploit the Unknown Gradually: One-Shot Video-Based Person Re-Identification by Stepwise Learning](https://yu-wu.net/pdf/CVPR2018_Exploit-Unknown-Gradually.pdf).


### Dependencies
- Python 3.6
- PyTorch (version >= 0.2.0)
- h5py, scikit-learn, metric-learn, tqdm

### Run
Move the downloaded dataset file `dukemtmc_videoReID.zip` to `./data/` and unzip here.

```shell
python3 run.py --dataset dukemtmc_videoReID --logs_dir logs/dukemtmc_videoReID_baseline/ --max_frames 900 --batch_size 16
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
Please refer to the license file for [DukeMTMC-VideoReID](https://github.com/Yu-Wu/DukeMTMC-VideoReID/blob/master/LICENSES/LICENSE_DukeMTMC-VideoReID.txt) and [DukeMTMC](https://github.com/Yu-Wu/DukeMTMC-VideoReID/blob/master/LICENSES/LICENSE_DukeMTMC.txt).


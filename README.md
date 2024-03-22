# Self-Supervised Denoising of Optical Coherence Tomography with Inter-Frame Representation

This paper was published in the proceedings of [ICIP2023]([https://arxiv.org/abs/2401.01548](https://ieeexplore.ieee.org/abstract/document/10223125))

## Abstract
Spectral-domain optical coherence tomography (SD-OCT) is a high-speed ocular imaging technology that is commonly employed in eye examinations to visualize the back structures of the eyes. OCT volume containing a sequence of cross-sectional images can be captured in seconds. However, the low signal-to-noise ratio (SNR) prevents accurate result interpretation. To obtain a high SNR OCT volume, numerous images must be averaged at each imaging depth, which is time-consuming. Subjects, especially children, who have short attention spans, may significantly hinder the data collection procedure. Most of the current algorithms focus on single-frame processing without using inter-frame information. Here we developed a lightweight 3D-UNet with a self-supervised strategy to denoise the low SNR OCT volume. This method does not require noisy-clean pairs and can be accomplished by simply measuring a volume containing multiple OCT images. The proposed method improves image quality with structural details preserved and achieves state-of-the-art performance on real OCT datasets.

## Results
![image](./fig/comparison_single.gif)

![image](./fig/comparison_volume.gif)

## Citation
If this work is useful for your research, please kindly cite it:
```
@inproceedings{
  author={Liu, Zhengji and Law, Tsz-Kin and Li, Jizhou and To, Chi-Ho and Chun, Rachel Ka-Man},
  booktitle={2023 IEEE International Conference on Image Processing (ICIP)}, 
  title={Self-Supervised Denoising of Optical Coherence Tomography with Inter-Frame Representation}, 
  year={2023},
  pages={3334-3338},
  doi={10.1109/ICIP49359.2023.10223125}}
```

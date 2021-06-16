# CFNet
This is the tensorflow implementation of [CFNet: Complementary Fusion Network for Rotation-Invariant Point Cloud Completion]().
## Environment Setup
We implement CFNet on tensorflow 1.12.0 with cuda9.0 and cudnn7.6
- Follow the steps in requirement.txt to install the necessary packages and compile the tensorflow extensions 
## Test Pretrained Models
- Download the dataset [here](https://drive.google.com/drive/folders/1P_W1tz5Q4ZLapUifuOE4rFAZp6L1XTJz).
- Download the pretrained models [here](https://drive.google.com/drive/folders/1d9IY6tv8uz_YTVg-Oj5NQhk_s5gRaCcD?usp=sharing)
- Follow test.txt for different test settings. 
 Modify the data directories according to your environment.
## Training
    python train.py
## Cite this work
    TODO
## Acknowledgements
This work borrows code from the following repositories
- [PCN](https://github.com/wentaoyuan/pcn)
- [Detail Preserved Point Cloud Completion via Separated Feature Aggregation](https://github.com/XLechter/Detail-Preserved-Point-Cloud-Completion-via-SFA)
- [RIConv](https://github.com/hkust-vgd/riconv)

### Instruction

1. Install [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

1. Convert a pre-trained MoCo model to detectron2's format:
   ```
   python3 convert-pretrain-to-detectron2.py input.pth.tar output.pkl
   ```

1. Put dataset under "./datasets" directory,
   following the [directory structure](https://github.com/facebookresearch/detectron2/tree/master/datasets)
	 requried by detectron2.

1. Run training:
   ```
   python train_net.py --config-file configs/pascal_voc_R_50_C4_24k_moco.yaml \
	--num-gpus 8 MODEL.WEIGHTS ./output.pkl
   ```

***Note:***  Variation on Pascal VOC is large: the std of AP50, AP, AP75 is expected to be 0.2, 0.2, 0.4 in most cases. We recommend to run 5 trials and compute means.

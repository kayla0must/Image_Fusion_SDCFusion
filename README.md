# SDCFuse: Scene Degradation-Aware Infrared-Visible Image Fusion in Extreme Conditions

This repository provides the official PyTorch implementation of the paper:

> **Scene Degradation-Aware Fusion Network for Robust Infrared and Visible Image Synthesis in Extreme Conditions**  
> Qingqing Hu, Yiran Peng, Zichun Shao, Kintak U, Junming Chen  
> *Submitted to* ***The Visual Computer***, 2025  
> 📄 [Project Page](https://github.com/kayla0must/Image-fusion)


The method consists of three major components:

1. **Scene Discriminator**: Uses a CLIP-based vision-language model to classify degradation types.
2. **Degradation Correction Units**: Integrates two lightweight modules:
   - **BANet** for brightness adjustment
   - **DFNet** for dehazing
3. **Fusion Backbone**: A multi-stage feature extraction and reconstruction network that merges corrected visible and infrared features.

SDCFuse is trained in **three stages** to progressively enhance the fusion quality with degradation-aware correction.


Repository Structure

```bash
├── Dataloader/            # Dataset loading and preprocessing
├── Model/                # Core model architecture: SDCFuse, BANet, DFNet
├── Train.py              # Fusion training script (Stage 2 & 3)
├── Train_CLS.py          # Scene classifier training script (Stage 1)
├── README.md             # This documentation

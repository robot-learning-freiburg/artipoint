# ArtiPoint + Arti4D
[**arXiv**](https://arxiv.org/abs/2509.01708) | [**Website**](https://artipoint.cs.uni-freiburg.de/) | [**Video**](https://youtu.be/uhd571Una-g?si=a4uO2oKJE2m8htH-)

This repository contains the implementation of ArtiPoint and the Arti4D dataset:
> **Articulated Object Estimation in the Wild**
>
> [Abdelrhman Werby]()&ast;, [Martin Büchner](https://rl.uni-freiburg.de/people/buechner)&ast;, [Adrian Röfer](https://rl.uni-freiburg.de/people/roefer)&ast;, [Chenguang Huang](https://www.utn.de/person/chenguang-huang/), [Wolfram Burgard](https://www.utn.de/person/wolfram-burgard-2/) and [Abhinav Valada](https://rl.uni-freiburg.de/people/valada). <br>
> &ast;Equal contribution. <br> 
>
> Conference on Robot Learning (CoRL), 2025.

<p align="center">
   <img src="./assets/artipoint-teaser.png" alt="Teaser of ArtiPoint and Arti4D" width="800" />
</p>


## 🏇 Quick Start

Follow these steps to install and run ArtiPoint.

1) Install environment

```bash
git clone https://github.com/robot-learning-freiburg/artipoint.git
cd artipoint
conda env create -f environment.yaml
conda activate artipoint
pip install -e .
```

2) Download model weights

```bash
mkdir -p checkpoints && cd checkpoints

# Mobile-SAM
gdown --fuzzy "https://drive.google.com/file/d/1dE-YAG-1mFCBmao2rHDp0n-PP4eH7SjE/view?usp=sharing"
unzip weight.zip

# CoTracker (offline and/or online)
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth || true

cd ..
```

3) Run on a dataset

```bash
# Minimal run (uses defaults)
artipoint

# Specify dataset root (Arti4D path)
artipoint dataset.root_path=/path/to/arti4d/raw

# Common options
artipoint \
   dataset.root_path=/path/to/arti4d/raw \
   debugging.write_results=true \
   visualization.show_3d_tracks=true
```


## 🗄️  Arti4D Dataset

Please choose a desired top-level folder, download, and unzip the following scene-wise splits as well as the metadata:
```
wget http://aisdatasets.cs.uni-freiburg.de/arti4d/arti4d-din080.zip
wget http://aisdatasets.cs.uni-freiburg.de/arti4d/arti4d-rh078.zip
wget http://aisdatasets.cs.uni-freiburg.de/arti4d/arti4d-rh201.zip
wget http://aisdatasets.cs.uni-freiburg.de/arti4d/arti4d-rr080.zip
wget http://aisdatasets.cs.uni-freiburg.de/arti4d/arti4d-meta.zip
```

Dataset Details:
- The Arti4D splits are named `din080`, `rh078`, `rh201`, `rr080`. For  each scene split you will find 8 to 16 sequences.
- The overall directory structure follows this format: `arti4d/raw/SCENE/SEQUENCE/`. Each scene defines a certain environment whereas the sequences are recordings within that particular scene.
- Difficulty levels and axis types  are contained in `arti4d/raw/metadata.yaml`.
- Within each sequence folder you will find:
      - Depth and RGB data under `depth` / `rgb`.
      - The interaction intervals are defined in `matched_cues.csv`.
      - The GT camera odometry is provided under `odom`.
      - We also provide a mesh and point cloud reconstructions generated via TSDF-fusion: `compressed_mesh.ply` / `compressed_point_cloud.ply`
      - Furthermore, there is also a compiled video of the demonstration sequence contained.

At the moment we do only support the raw data download. Please contact [buechner@cs.uni-freiburg.de](mailto:buechner@cs.uni-freiburg.de) in case you require the rosbag data.

**Predicted camera odometry**
Furthermore, we also provide the camera odometry of [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) and its registrations to the recorded maps using [KISS-MATCHER](https://github.com/MIT-SPARK/KISS-Matcher):
```
wget http://aisdatasets.cs.uni-freiburg.de/arti4d/droid-slam-outputs.zip
wget http://aisdatasets.cs.uni-freiburg.de/arti4d/droid-slam-registrations.zip
```

If you find our work useful, please consider citing our paper:
```
@article{werby2025articulated,
   author={Werby, Abdelrhman and Buechner, Martin and Roefer, Adrian and Huang, Chenguang and Burgard, Wolfram and Valada, Abhinav},
   title={Articulated Object Estimation in the Wild},
   journal={Conference on Robot Learning (CoRL)},
   year={2025},
}
```

## ToDo
- [x] Initial code release
- [ ] Evaluation protocol
- [ ] RIPL factor graph implementation (Buchanan et al.)


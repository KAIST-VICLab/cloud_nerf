# Cloud NeRF
This is the official implementation of Neural Radiance Fields with Points Cloud Latent Representation.
[[Paper]]()[[Project Page]]()
## Instruction
- Please download and arrange the dataset same as the instruction.
- For the environment, we provide our Docker image for the best reproduction.
- The scen optimization scripts are provided in the instruction.
## Data
- We evaluate our framework on forward-facing LLFF dataset, available at [Google drive]('https://drive.google.com/drive/folders/14boI-o5hGO9srnWaaogTU5_ji7wkX2S7').
- We also need to download the pre-trained MVS depth estimation at [Google drive]('https://drive.google.com/drive/folders/13lreojzboR7X7voJ1q8JduvWDdzyrwRe').
- Our data folder structure is same as follow:
```

├── datasets
│   ├── nerf_llff_data
    │   │   │──fern
    │   │   │  |   |──depths
    │   │   │  |   |──iamges
    │   │   │  |   |──images_4
    │   │   │  |   |──sparse
    │   │   │  |   |──colmap_depth.npy
    │   │   │  |   |──poses_bounds.npy
    │   │   │  |   |──...
```
## Docker
- We provide the Docker images of our environment at [DockerHub]('https://hub.docker.com/repository/docker/quan5609/cloud_nerf').
- To create docker container from image, run the following command
  ```
  docker run \
  --name ${CONTAINER_NAME} \
  --gpus all \
  --mount type=bind,source="${PATH_TO_SOURCE}",target="/workspace/source" \
  --mount type=bind,source="${PATH_TO_DATASETS}",target="/workspace/datasets/" \
  --shm-size=16GB \
  -it ${IMAGE_NAME}
  ```
## Train & Evaluation
- To train from scratch, run the following command
  ```
  CUDA_VISIBLE_DEVICES=1 python train.py \
  --dataset_name llff \
  --root_dir /workspace/datasets/nerf_llff_data/${SCENE_NAME}/ \
  --N_importance 64 \
  --N_sample 64 \
  --img_wh 1008 756 \
  --num_epochs 10 \
  --batch_size 4096 \
  --optimizer adam \
  --lr 5e-3 \
  --lr_scheduler steplr \
  --decay_step 2 4 6 8 \
  --decay_gamma 0.5 \
  --exp_name ${EXP_NAME}
  ```
- To evaluate a checkpoint, run the following command
  ```
  CUDA_VISIBLE_DEVICES=1 python eval.py \
  --dataset_name llff \
  --root_dir /workspace/datasets/nerf_llff_data/${SCENE_NAME}/ \
  --N_importance 64 \
  --N_sample 64 \
  --img_wh 1008 756 \
  --weight_path ${PATH_TO_CHECKPOINT} \
  --split val
  ```
## Visualization
- Visualization of Fern scene
<img src="assets/fern.gif" width="400">
- Visualization of Horn scene
<img src="assets/horn.gif" width="400">
- Visualization of Room scene
<img src="assets/room.gif" width="400">
# Acknowledgement
Our repo is based on [nerf]('https://github.com/bmild/nerf'), [nerf_pl]('https://github.com/kwea123/nerf_pl'), [DCCDIF]('https://github.com/lity20/DCCDIF'), and [Pointnet2_PyTorch]('https://github.com/erikwijmans/Pointnet2_PyTorch').
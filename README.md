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
- We provide the Docker images of our environment at [DockerHub]('https://drive.google.com/drive/folders/14boI-o5hGO9srnWaaogTU5_ji7wkX2S7').
- To create docker container from image run the following command
  ```
  docker run \
  --name ${CONTAINER_NAME} \
  --gpus all \
  --mount type=bind,source="${PATH_TO_SOURCE}",target="/workspace/source" \
  --mount type=bind,source="${PATH_TO_DATASETS}",target="/workspace/datasets/" \
  --shm-size=16GB \
  -it ${IMAGE_NAME}
  ```

## Visualization
# Acknowledgement
Our repo is based on [nerf](https://github.com/bmild/nerf), [nerf_pl](https://github.com/kwea123/nerf_pl), and [DCCDIF](https://github.com/lity20/DCCDIF)
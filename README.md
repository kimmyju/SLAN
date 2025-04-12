# Light Awareness Network for Shadow Detection (ICNGCT 2023)

**SLAN (Shadow Light Awareness Network)** is a **U-Net** based shadow detection model designed to address various distortions that may potentially degrade the performance of segmentation tasks, by utilizing light source and object information. By introducing Light-source Object Information (LOI), SLAN effectively localizes shadow regions. It shows robust performance on the SOBA dataset, confirming its effectiveness in challenging scenarios.

This work was presented as an **oral presentation** at the *2023 International Conference on Next Generation Computing and Communication Technologies (ICNGCT)*.

> ★ If you found this repository useful, please support it by giving a ⭐!

## 📦 Requirements

Make sure the following environments and packages are installed:

- Ubuntu 22.04
- CUDA 11.6
- Python 3.9
- Pytorch 1.13.1

---

## ⚙️ Environment Setup 

### 1. Clone the Repository

```bash
git clone https://github.com/kimmyju/SLAN.git
```
### 2. Create Conda Environment

Create a new Conda environment with Python 3.9:

```bash
conda create -n <your_env_name> python=3.9
```
### 3. Install Dependencies

You can install the required packages with the following command:

```bash
pip install -r requirements.txt
```

### 4. Prepare model file and SOBA Dataset

We converted the SOBA (Shadow-Object Association) dataset originally introduced in the CVPR 2020 paper [*Instance Shadow Detection*](https://github.com/stevewongv/InstanceShadowDetection) into a semantic segmentation format suitable for our task. The final dataset consists of 1,080 training images and 30 testing images. You can download the converted SOBA dataset and original paper from [Google Drive](https://drive.google.com/drive/folders/1N5W6UsecEBteExKbKDzAPg9LIFUd6Y1p?usp=sharing).

After downloading, place the `data` folder in the root directory of this project.
This folder should contain the following items:

- `imgs/`
- `object_masks/`
- `shadow_masks/`
- `light_annotations.txt`
- `SBU-test`

The final data structure should look like this:

```bash
.
└── checkpoint/
    ├── imgs
    ├── object_masks
    ├── shadow_masks
    ├── light_annotations.txt
    └── SBU-test/
        ├── test_imgs
        ├── test_object_masks
        ├── test_shadow_masks
        └── test_light_annotations.txt
```

> ✅ Due to limited data, the training set was used exclusively for training, while the SBU-test set was used for validation only. Validation images do not affect the model, as they are used only to monitor performance and have no impact on weight updates.

---

## 🧪 Demo

To run the demo, you can simply run:

```bash
sh demo.sh
```

> ✅ make sure to specify the **model path**, **image name**, **output file name**, and **light direction index** inside the script.

---

## 📈 Train the Model

To train the model, run:

```bash
sh train.sh
```   

**Parameter Descriptions**

In this script, you must specify both the number of epochs and the batch size.
Additionally, the following parameters should also be defined:

- `--val_step`: Run validation every *n* epochs.  
- `--checkpoint_step`: Save model checkpoints every *n* epochs.

---

## Acknowledgements & References

This project uses or is based on the following resources:

- **U-Net** – [https://github.com/milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)  
  Used as architectural reference. Licensed under GPL v3.
- **SOBA Dataset** – [https://github.com/stevewongv/InstanceShadowDetection](https://github.com/stevewongv/InstanceShadowDetection)  
SOBA Dataset. Licensed under Apache License 2.0.

---

## License

This project is licensed under the **GNU General Public License v3.0** (see `LICENSE`).
It also includes third-party data licensed under the **Apache License 2.0**.  
For details, refer to the `third_party_licenses/` directory.

---

📄 Light Awareness Network for Shadow Detection

Originally proposed by Ye Ju Kim, Junyong Hong, *et al.*

![Network](https://github.com/user-attachments/assets/c4f49c3d-0672-4382-8025-b6ced93fb737)

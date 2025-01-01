# Image Inpainting for Irregular Holes with Partial Convolution

This repository contains the PyTorch implementation of the paper **"Image Inpainting for Irregular Holes with Partial Convolution."**

## Prerequisites
- This code assumes input images are in PNG format and have a resolution of 256x256 pixels.
- If you wish to use images with different resolutions or formats, modify the `main.py` and the `get_loader` method in `dataset.py` accordingly.
  ```
  # main.py
  
  def main(config):
    set_seed(58)
    train_imgs = os.path.join(config.train_dir, "*.png") # <-- edit this!
    train_imgs = glob.glob(train_imgs)
    valid_imgs = os.path.join(config.valid_dir, "*.png") # <-- edit this!
    valid_imgs = glob.glob(valid_imgs)
    ...
  ```
  ```
  # dataset.py
  
  def get_loader(img_paths, batch_size: int, shuffle: bool):
      transform = transforms.Compose([
          transforms.Resize((256, 256)), # <-- edit this!
          transforms.ToTensor(), 
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])
      dataset = ImageDataset(img_paths, transform)
      return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
  ```

## How to Use
1. Create a `config.yaml` file based on the provided `config.yaml.template`.
2. Run `main.py`.
    ```
    main.py --conf config.yaml
    ```
   - The results will be logged to TensorBoard.

## Experimental Results
<img src="https://github.com/user-attachments/assets/2015ab0e-9c0b-4979-828f-38de8c71d79d" width="300" height="400"/>
<img src="https://github.com/user-attachments/assets/a9c87c29-349d-4a9b-b69e-e59c65240640" width="650" height="400"/>

The image above shows the output after training for 5 epochs using approximately 30,000 images of size 256x256.  
The results are not fully converged; therefore, longer training and a larger dataset can lead to improved outcomes.

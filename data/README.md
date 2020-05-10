# About the data
The data has been created using just 200 images! Later sections will explain the process in detail.
## Contents
As a part of the dataset, we have:

| Name | No. of Images | Description |
| :--: | :-----------: | :---------: |
| bg | 100 | Different street scenes, used as background. |
| fg | 100 | Different footballers' images, without any background. | 
| fg_mask | 100 | Mask of each of the foreground image. |
| fg_bg | 400k | Foreground images layed over backgrounds, at random locations. |
| fg_bg_mask | 400k | Masks, each corresponding to one fg_bg image. |
| fg_bg_depth | 400k | Depth maps, each corresponding to one fg_bg image. |

Note: Since it wasn't feasible to capture depth maps using depth cameras, they have been created using <a href="https://github.com/ialhashim/DenseDepth/blob/master/DenseDepth.ipynb" target="_blank">`Depth Model`</a>.

## Samples
> bg
![bg](Samples/bg.jpg)
> fg
![bg](Samples/fg.jpg)
> fg_mask
![bg](Samples/fg_mask.jpg)
> fg_bg
![bg](Samples/fg_bg.jpg)
> fg_bg_mask
![bg](Samples/fg_bg_mask.jpg)
> fg_bg_depth
![bg](Samples/fg_bg_depth.jpg)

## Link to the dataset
The data has been divided into three parts and stored in zip format. Zip files are easy to extract, move or download, since they are considered as a single file (rather than multiple images in this case).

| Content | Size (Approx) | No. of Images | Link |
| :-----: | :--: | :-----------: | :--: |
| bg, fg, fg_mask | 21 MB | 300 | https://drive.google.com/open?id=1eMjj_oWB5jEq4jhd2l4kKE3PL_vf4PZT |
| fg_bg, fg_bg_mask | 4 GB | 800k | https://drive.google.com/open?id=1TxFhTFP-pUSBtywjshm1sHpPUxmVR7Gc | 
| fg_bg_depth | 1 GB | 400k | https://drive.google.com/open?id=1CFjwwnY23u2UCzuzZWF03xkEDOhoKBqJ |

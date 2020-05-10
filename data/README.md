# About the data
## Contents
As a part of the dataset, we have:

| Name | No. of Images | Description |
| :--: | :-----------: | :---------: |
| bg | 100 | Different street scenes, used as background. |
| fg | 100 | Different footballers' images, without any background. | 
| fg_mask | 100 | Mask of each of the foreground image. |
| fg_bg | 400k | Foreground images layed over backgrounds, at random locations.
| fg_bg_mask | 400k | Masks, each corresponding to one fg_bg image. |
| fg_bg_depth | 400k | Depth maps, each corresponding to one fg_bg image. |

Note: Since it wasn't feasible to capture depth maps using depth cameras, they have been created using <a href="https://github.com/ialhashim/DenseDepth/blob/master/DenseDepth.ipynb" target="_blank">`Depth Model`</a>.

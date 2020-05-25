# About the data
The data has been created using just 200 images! To keep the size of the dataset small, the images have been saved in `.jpg` format rather than `.png`. Later sections will explain the process of creation in detail.
## Contents
As a part of the dataset, we have:

| Name | No. of Images | Image dimensions | Description |
| :--: | :-----------: | :--------------: | :---------: |
| bg | 100 | 224 x 224 x 3 | Different street scenes, used as background. |
| fg | 100 | variable, 4 channels | Different footballers' images, without any background. | 
| fg_mask | 100 | variable, 1 channel | Mask of each of the foreground image. |
| fg_bg | 400k | 224 x 224 x 3 | Foreground images layed over backgrounds, at random locations. |
| fg_bg_mask | 400k | 224 x 224 x 1 | Masks, each corresponding to one fg_bg image. |
| fg_bg_depth | 400k | 224 x 224 x 1 | Depth maps, each corresponding to one fg_bg image. |

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
| bg, fg, fg_mask | 21 MB | 300 |  <a href="https://drive.google.com/open?id=1eMjj_oWB5jEq4jhd2l4kKE3PL_vf4PZT" target="_blank">Link</a> |
| fg_bg, fg_bg_mask | 4 GB | 800k | <a href="https://drive.google.com/open?id=1TxFhTFP-pUSBtywjshm1sHpPUxmVR7Gc" target="_blank">Link</a>  | 
| fg_bg_depth | 1 GB | 400k | <a href="https://drive.google.com/open?id=1CFjwwnY23u2UCzuzZWF03xkEDOhoKBqJ" target="_blank">Link</a> |

## Mean and STD
You can refer to this <a href="https://github.com/akshatjaipuria/Mask-and-Depth-Prediction/blob/master/data/mena_std_calculation.ipynb" target="_blank">`Notebook`</a> for the script to calculate the mean and standard deviation of the dataset. PyTorch's Dataloader was used to load images in batches and perform the calculations. Note that the values are for the standardized pixel values, between 0-1.

| Name | No. of Channels | Mean | STD |
| :--: | :-------------: | :--: | :-: |
| bg | 3 | [0.5039, 0.5001, 0.4849] | [0.2465, 0.2463, 0.2582] |
| fg_bg | 3 | [0.5057, 0.4966, 0.4812] | [0.2494, 0.2498, 0.2612] |
| fg_bg_mask | 1 | [0.0498] | [0.2154] |
| fg_bg_depth | 1 | [0.4373] | [0.2728] |

## Directory structure and organization
The directory structures after extracting the three zip files are as follows:
> File 1
```
dataset.
|
+---bg
|       bg_001.jpg
|       ...
|       bg_100.jpg
|       
+---fg
|       fg_001.png
|       ...
|       fg_100.png
|       
\---fg_mask
        fg_mask_001.jpg
        ...
        fg_mask_100.jpg
```
> File 2
```
data.
|
+---fg_bg
|   +---bg_001
|   |       img_0001.jpg
|   |       ...
|   |       img_4000.jpg
|   +---...
|   \---bg_100
|
\---fg_bg_mask
    +---bg_001
    |       img_0001.jpg
    |       ...
    |       img_4000.jpg
    +---...
    \---bg_100
```
> File 3
```
fg_bg_depth.
|
+---bg_001
|       img_0001.jpg
|       ...
|       img_4000.jpg
+---...
\---bg_100
```
## How were they created?

### bg
I collected 100 street images, preferably empty, having dimensions more than 250. These images were then square cropped and saved. They were scaled down to 224 x 224 with the help of simple script using OpenCV function `cv2.resize()`.

### fg
For fg, I collected 100 images of footballers. Of these images, the backgrounds were removed using bacground removal tool available in `Microsoft PowerPoint`. Since the background region were made transparent, an extra channel, alpha channel, was added to the image and hence they had to be saved in PNG format.

### fg_mask
These were created using the alpha channel of the fg images. Alpha channel contains pixel value 0 for completely transparent, 255 for completely opaque and Values in between 0-255 represent partially transparent. I copied the alpha channel as a one channel image, and changed all the non zero pixel values to 255. This single channel is our mask. Refer to this <a href= "https://github.com/akshatjaipuria/Mask-and-Depth-Prediction/blob/master/data/fg_mask.ipynb" target="_blank">`Notebook`</a> for the code.

### fg_bg and fg_bg_mask
These two images were created simultaneously. For fg_bg, on each of the 100 bg images, a single fg was placed at 20 different random locations and the same fg was flipped and again placed at 20 different random locations. It was repeated for all 100 images over each background image. This way, we ended up with 100*(100*(20+20)) = 400k images. Before placing fg over bg, the fg had to be scaled down to fit in the bg, and since height was the deciding factor, I made it constant, 105, for all the fgs and the width of each image was calculated according to its aspect ration in order to retain information. Random x and y was generated keeping max_x and max_y as the difference between the dimensions of bg and fg so that fg dosen't go outside the bg. 

For placing fg over bg, alpha channel of fg was used as the medium to decide which pixels had to be manipulated. At the corresponding random loaction on the bg, the pixel region on the bg of size equal to the fg was altered as, if the alpha channel pixel of fg was 0, bg pixel was left as it is and if the alpha channel pixel was non zero, bg pixel was replaced with fg pixel. This way, we had the fg placed over bg.

For fg_bg_mask, a black image of 224 x 224 was generated simultaneously just after one fg was placed over one bg and the pixels of this black image was altered in a similar way to create mask as, if the alpha channel pixel of fg was 0, bg pixel was left as it is and if the alpha channel pixel was non zero, bg pixel was replaced with 255 (white).

Refer to this <a href= "https://github.com/akshatjaipuria/Mask-and-Depth-Prediction/blob/master/data/data_creation.ipynb" target="_blank">`Notebook`</a> for the code.

### fg_bg_depth
This was created using <a href="https://github.com/ialhashim/DenseDepth/blob/master/DenseDepth.ipynb" target="_blank">`Depth Model`</a>.
The output dimensions of dense depth was half of the input and the model didn't work well with smaller images. Due to this reason, I modified the load_image function to accept 500 images at a time and scale them up to 448 x 448 from 224 x 224. This way, the depth model was able to perform well on the inputs as well as we got our desired dimensions of 224 x 224 as output. The code implementation to generate the images is available in this <a href= "https://github.com/akshatjaipuria/Mask-and-Depth-Prediction/blob/master/data/Depth_Model.ipynb" target="_blank">`Notebook`</a>.

## Size and Speed Management
To keep the size of the dataset small, the images were saved in JPG format rather than PNG. Wherever possible, the channels were kept 1 insted of 3. Also, while creating fg_bg and fg_bg_mask, the images were saved with quality as 65% to keep the file size low.

To achive a good speed while creating data, instead of reading images again and again while creating fg_bg, the copies were made whenever required in the code itself to save read operation and thus time. Copies of fg amd bg were made on the go.
Further improvements in the speed could have been done by further reducing the read operations by storing all 100 bg and 100 fg in the code itself.

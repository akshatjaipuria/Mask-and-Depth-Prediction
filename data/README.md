## About the data
As a part of the dataset, we have:

> For Input
- bg - 100 different background images.
- fg_bg - 400k foregroung images layed over (100) backgrounds.

> For Output
- fg_bg_mask - 400k masks, each corresponding to one fg_bg image.
- fg_bg_depth - 400k depth maps, each corresponding to one fg_bg image.

Note: Since it wasn't feasible to capture depth maps using depth cameras, they have been created using <a href="https://github.com/ialhashim/DenseDepth/blob/master/DenseDepth.ipynb" target="_blank">`Depth Model`</a>.

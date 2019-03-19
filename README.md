# Foreground-detection
This program uses breadth first search to detect the foreground of an image. It works best when there are clear changes in color at the boundary of the foreground object, and sometimes misreads shadows as part of the foreground. It can sometimes detect complex foreground images on complex backgrounds, however. On a test set of ten images, it did reasonably well on the following seven: 

![Arguably successful outcomes from the test set](https://imgur.com/229Khds.jpg)

However, it did not do as well with the following three images: 

![Unsuccessful outcomes from the test set](https://imgur.com/gPKv5Kh.jpg)

## How it works

### Breadth first search
The program first assumes that the border pixels in an image are a part of the background. Then it performs a breadth first search, beginning with those border pixels, and working its way inward. When considering a neighbor of a background pixel, the program examines the difference in color between the two pixels, and calculates a number meant to represent the probability that two such colors would appear in adjacent pixels on the same object. If this probability is above a certain cutoff, then the neighbor pixel is included in the list of background pixels. 

### Layering and Gaussian blurring
To reduce the number of computations and to ignore subtle edges within the background, we begin with a very low resolution version of the image, in which the shorter side is at most 40 pixels. We perform the above procedure on this low resolution image, and label its pixels as foreground or background accordingly. Then we resize the image containing the labelling (scaling up by a factor of 1.5), blur it, and create a new list of background pixels for the larger version of the image based on this blurred version. 

We repeat this process, gradually refining our labeling on higher and higher resolution versions of the image, until we arrive at an image with a longer side of 400 pixels. 

### Hue/saturation/value metric

### Gradient descent and hyperbola fitting

### Largest connected component

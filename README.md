# Foreground-detection
This program uses breadth first search to detect the foreground of an image. It works best when there are clear changes in color at the boundary of the foreground object, and sometimes misreads shadows as part of the foreground. It can sometimes detect complex foreground images on complex backgrounds, however. On a test set of ten images, it did reasonably well on the following seven: 

![Arguably successful outcomes from the test set](https://imgur.com/229Khds.jpg)

However, it did not do as well with the following three images: 

![Unsuccessful outcomes from the test set](https://imgur.com/gPKv5Kh.jpg)

## How it works

### Breadth first search
The program first assumes that the border pixels in an image are a part of the background. Then it performs a breadth first search, beginning with those boundary pixels, and working its way inward. When considering a neighbor of a background pixel, the program examines the difference in color between the two pixels, and calculates a number meant to represent the probability that two such colors would appear in adjacent pixels on the same object. If this probability is above a certain cutoff, then the neighbor pixel is included in the list of background pixels. 

### Layering and Gaussian blurring

### Hue/saturation/value metric

### Gradient descent and hyperbola fitting

### Largest connected component

# Foreground detection
This program uses breadth first search to detect the foreground of an image. It works best when there are clear changes in color at the boundary of the foreground object, and sometimes misreads shadows as part of the foreground. It can sometimes detect complex foreground images on complex backgrounds, however. On a test set of ten images, it did reasonably well on the following seven: 

![Arguably successful outcomes from the test set](https://imgur.com/229Khds.jpg)

However, it did not do as well with the following three images: 

![Unsuccessful outcomes from the test set](https://imgur.com/gPKv5Kh.jpg)

## How it works

### Breadth first search
The program first assumes that the border pixels in an image are a part of the background. Then it performs a breadth first search, beginning with those border pixels, and working its way inward. When considering a neighbor of a background pixel the program examines the difference in color between the two pixels, and calculates a number meant to represent the probability that two such colors would appear in adjacent pixels on the same object. If this probability is above a certain cutoff, then the neighbor pixel is included in the list of background pixels. 

### Layering and Gaussian blurring
To reduce the number of computations and to ignore subtle edges within the background, we begin with a very low resolution version of the image, in which the shorter side is at most 40 pixels. We perform the above procedure on this low resolution image, and label its pixels as foreground or background accordingly. Then we resize the image containing the labelling (scaling up by a factor of 1.5), apply a Gaussian blur, and create a new list of background pixels for the larger version of the image based on this blurred version. 

We repeat this process, gradually refining our labeling on higher and higher resolution versions of the image, until we arrive at an image with a longer side of 400 pixels. 

This image shows this process of gradual refinement for a picture of a pear: 

![Layering a pear](https://i.imgur.com/LdnnSZM.jpg)

### Hue/saturation/value metric
To approximate the chance that two colors on adjacent pixels belong to the same object, we first find their differences in hue, saturation, and value. The logic in using HSV rather than RGB is that a change in hue may be more likely to indicate a boundary between two objects, while a change in saturation or value may indicate the beginning on a shadow on a single object, so it may be helpful to consider hue separately. 

Note that HSV color space uses a non-Euclidean metric, since fixing saturation and value but allowing hue to vary results in a circle. 

### Gradient descent and hyperbola fitting
To accomplish the above we must determine appropriate cutoffs for differences in hue, saturation, and value for the image. To do this, we look at the distribution of differences between these values for adjacent pixels. These distributions tends to resemble hyperbolas of the form (1 + (m*x)^2)^0.5 - m*x (after scaling both dimensions appropriately). So, we use gradient descent to determine the value of m which best approximates the distribution for each of the three color parameters. We then place greater emphasis on differences in hue by doubling the value of m associated to hue. 

This image shows these three distributions for an image, along with the three hyperbolas fit to them: 

![Modeling HSV differences for an image](https://i.imgur.com/yooCa3K.jpg)

Note that this fit is imperfect. In future work we may fit a hyperbola of the form n((m*x)^2)^0.5 - m*x) instead, and use two dimensional gradient descent (on n and m) to improve the fit. 

### Largest connected component
Once we have labelled all pixels in the highest resolution image as foreground or background, we use breadth first search to determine the largest connected component of the foreground, and eliminate all pixels classified as foreground outside of this component. 

![The largest connected component of the foreground pixels for a pear image](https://i.imgur.com/zWXAA5t.jpg)

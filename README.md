# Foreground-detection
This program uses breadth first search to detect the foreground of an image. It works best when there are clear changes in color at the boundary of the foreground object, and sometimes misreads shadows as part of the foreground. It can sometimes detect complex foreground images on complex backgrounds, however. On a test set of ten images, it did reasonably well on the following seven: 

![Arguably successful outcomes from the test set](https://imgur.com/229Khds)

However, it did not do as well with the following three images: 

![Unsuccessful outcomes from the test set](https://imgur.com/gPKv5Kh)

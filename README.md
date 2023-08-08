# Fracture Suite

This package helps identifying splinters on broken glass plys.

It performs several operations on the input image to enhance the visibility of scanned cracks and analyzes contours in the image. Filtering then helps to remove unwanted artifacts like dust speckles or glue residue. The remaining contours are then used to calculate the size (in px) as well as the round- and rough-ness of the splinter.

![Backend plot of analyzer, displaying original and preprocessed image and detected cracks](.content/backend.png)

## How it works

Several steps are necessary to analyze a fracture scan:
1. Cropping of input image (_optional_)
   1. Analyze the image and find the biggest rectangular shape
   2. Perspective transform the image, so that the rectangle is filling the extents
2. Preprocessing
   1. Gaussian Blur + (_optional_) Resize of the input image
   2. Adaptive Threshold
3. Contour detection
   1. Find all contours on the preprocessed image
   2. Filter Contours, remove all:
      1. Whose perimeter is too small
      2. Whose area is way too large (25000px²)
4. Create stencil with the found contours
   1. This helps to quickly remove all contours that lie within a bigger contour
   2. Draw all contours onto a new image (resulting image will display the cracks)
5. Skeletonization #1
   1. Skeletonize the stencil to minimize the crack width to 1px wide lines
   2. Use Erode/Dilate (closing kernel) to connect gaps in contours (this will widen the 1px wide lines)
6. Skeletonization #2
   1. Skeletonize the image again to retrieve the crack middle lines
7. Contour detection #2
   1. Now with minimal fuzziness, run the splinter detection again
8. Create splinters from resulting contour list

## Installation

```bat
pip install fracsuite
```

## Usage

For details see: [API Docs](fracsuite.md)

### Use the module directly

```bat
py -m fracsuite.splinters -image "path/to/image" [--crop]
```

#### `-image`

The path to the image

#### `--crop`

If the image contains unfiltered area around the ply, use this to crop the image to the ply.

### Create a script

```python
from fracsuite.splinters.analyzer import Analyzer

image = r"Path/to/some/image.bmp"
crop = True

analyzer = Analyzer(image, crop)

analyzer.plot()
analyzer.plot_area()
analyzer.plot_area_2()
```
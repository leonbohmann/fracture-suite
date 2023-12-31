# Table of Contents

* [analyzer](#analyzer)
  * [crop\_perspective](#analyzer.crop_perspective)
  * [preprocess\_image](#analyzer.preprocess_image)
  * [filter\_contours](#analyzer.filter_contours)
  * [Analyzer](#analyzer.Analyzer)
    * [plot](#analyzer.Analyzer.plot)
    * [plot\_area](#analyzer.Analyzer.plot_area)
    * [plot\_area\_2](#analyzer.Analyzer.plot_area_2)
* [splinter](#splinter)
  * [Splinter](#splinter.Splinter)
    * [calculate\_roughness](#splinter.Splinter.calculate_roughness)
* [\_\_init\_\_](#__init__)
* [\_\_main\_\_](#__main__)

<a id="analyzer"></a>

# analyzer

<a id="analyzer.crop_perspective"></a>

#### crop\_perspective

```python
def crop_perspective(img, size=4000, dbg=False)
```

Crops a given image to its containing pane bounds. Finds smallest pane countour with
4 corner points and aligns, rotates and scales the pane to fit a resulting image.

**Arguments**:

- `img` _Image_ - Input image with a clearly visible glass pane.
- `size` _int_ - Size of the resulting image after perspective transformation.
- `dbg` _bool_ - Displays debug image outputs during processing.
  

**Returns**:

- `img` - A cropped image which only contains the glass pane. Size: 1000x1000.
  If no 4 point contour is found, the whole image is returned.

<a id="analyzer.preprocess_image"></a>

#### preprocess\_image

```python
def preprocess_image(image,
                     gauss_sz=(3, 3),
                     gauss_sig=5,
                     block_size=11,
                     C=6,
                     rsz=1) -> nptyp.ArrayLike
```

Preprocess a raw image.

**Arguments**:

- `image` _nd.array_ - The input image.
- `gauss_sz` _size-tuple, optional_ - The size of the gaussian filter.
  Defaults to (5,5).
- `gauss_sig` _int, optional_ - The sigma for gaussian filter. Defaults to 3.
- `block_size` _int, optional_ - Block size of adaptive threshold. Defaults to 11.
- `C` _int, optional_ - Sensitivity of adaptive threshold. Defaults to 6.
- `rsz` _int, optional_ - Resize factor for the image. Defaults to 1.
  

**Returns**:

- `np.array` - Preprocessed image.

<a id="analyzer.filter_contours"></a>

#### filter\_contours

```python
def filter_contours(contours, hierarchy) -> list[nptyp.ArrayLike]
```

This function filters a list of contours.

<a id="analyzer.Analyzer"></a>

## Analyzer Objects

```python
class Analyzer(object)
```

Analyzer class that can handle an input image.

<a id="analyzer.Analyzer.plot"></a>

#### plot

```python
def plot(region=None) -> None
```

Plots the analyzer backend.
Displays the original img, preprocessed img, and an overlay of the found cracks
side by side in a synchronized plot.

<a id="analyzer.Analyzer.plot_area"></a>

#### plot\_area

```python
def plot_area() -> Figure
```

Plots a graph of accumulated share of area for splinter sizes.

**Returns**:

- `Figure` - The figure, that is displayed.

<a id="analyzer.Analyzer.plot_area_2"></a>

#### plot\_area\_2

```python
def plot_area_2() -> Figure
```

Plots a graph of Splinter Size Distribution.

**Returns**:

- `Figure` - The figure, that is displayed.

<a id="splinter"></a>

# splinter

<a id="splinter.Splinter"></a>

## Splinter Objects

```python
class Splinter()
```

<a id="splinter.Splinter.calculate_roughness"></a>

#### calculate\_roughness

```python
def calculate_roughness() -> float
```

Calculate the roughness of the contour by comparing the circumfence
of the contour to the circumfence of its convex hull.

**Returns**:

- `float` - A value indicating how rough the perimeter is.

<a id="__init__"></a>

# \_\_init\_\_

<a id="__main__"></a>

# \_\_main\_\_


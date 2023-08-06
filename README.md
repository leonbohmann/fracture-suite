# Fracture Suite

This package helps identifying splinters on broken glass plys.

## Installation

```bat
pip install fracsuite
```

## Usage
```bat
py -m fracsuite -image "path/to/image" [--crop]
```

#### `-image`

The path to the image

#### `--crop`

If the image contains unfiltered area around the ply, use this to crop the image to the ply.
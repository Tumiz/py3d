# PY3D - 3D data analysis toolkit
[Code](https://github.com/Tumiz/py3d) | [Docs](https://tumiz.github.io/py3d)

## What is it?

**py3d** is a pure and lightweight Python library dedicated to 3D data structures and functions. It enables batch processing of 3D points, lines, planes, and 3D meshes. Moreover, it provides interactive visualization capabilities for these elements. It is advisable to use it in Jupyter for visualization purposes.

## Main features

* Supports the reading, writing, and visualization of multiple 3D data formats such as the popular PLY, NPY, CSV, PCD, and OBJ. [Demos](https://tumiz.github.io/py3d/IO.html)
* Perform rotation, translation, and scaling operations on 3D geometries. [Demos](https://tumiz.github.io/py3d/Transform.html)
* Conversion among diverse rotation representations, [Demos](https://tumiz.github.io/py3d/Rotation.html) 
    * Rotation matrix
    * Angle - axis
    * Quaternion
    * Euler angle
    * Rotation vector

## How to install it?
```
pip install py3d
```

## How to use it?

Here are some small examples:

1. Visualize a pcd file in jupyter


```python
import py3d
pcd = py3d.read_pcd("binary.pcd")
print("min", pcd.min())
print("max", pcd.max())
pcd.xyz.as_point(colormap=pcd.w)
```

2. Visualize an image in jupyter


```python
import py3d
py3d.read_img("./20220917214012.jpg").as_image()
```

3. Convert euler angles to rotation matrix 


```python
import py3d
py3d.Transform.from_euler("xyz", [0.4, -0.2, 0])
```

## API reference

[Vector](https://tumiz.github.io/py3d/Vector.html)
[Vector3](https://tumiz.github.io/py3d/Vector3.html)
[Vector4](https://tumiz.github.io/py3d/Vector4.html)
[Transform](https://tumiz.github.io/py3d/Transform.html)
[Rotation](https://tumiz.github.io/py3d/Rotation.html)
[Color](https://tumiz.github.io/py3d/Color.html)
[IO](https://tumiz.github.io/py3d/IO.html)

[Top](#py3d)



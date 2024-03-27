# Scenario Tool Kit
**Scenario** [Github](https://github.com/Tumiz/scenario) is a series of python libraries that helping building simulation environment easily and quickly. Click [Here](https://tumiz.github.io/scenario/) to see documents 

## py3d

**py3d** is one of the **scenario** libraries. It is a python 3d computational geometry library, which can deal with points, lines, planes and 3d meshes in batches.

It is under development and unstable currently. But it is designed to be simple, stable and customizable:

* simple means api will be less than usual and progressive
* stable means it will have less dependeces and modules, and it will be fully tested
* customizable means it will be a libaray rather than an application, it only provide data structures and functions handling basic geometry concepts

Now supported features includes:

* read/visualize ply
* read/write/visualize npy
* read/write/visualize csv
* read/write/visualize pcd
* apply rotation/translation/scaling on vectors
* compose or decompose different transforms
* conversion between rotation matrix, angle axis, quaternion, euler angle and rotation vector

### Installation
```
pip install py3d
```

### Example


```python
import py3d
import numpy
cars = py3d.cube(0.5,0.2,0.3) @ py3d.Transform.from_translation(y=range(1,6), z=0.15)
cars.paint()
t = 0
dt = 0.1
while t<4:
    py3d.render(cars, t=t)
    cars @= py3d.Transform.from_rpy(py3d.Vector3(z=dt * numpy.linspace(0.1,1,5)))
    t += dt
py3d.show()

```

![example](docs/index.gif)

### API reference

[Vector](https://tumiz.github.io/scenario/Vector.html)
[Vector3](https://tumiz.github.io/scenario/Vector3.html)
[Vector4](https://tumiz.github.io/scenario/Vector4.html)
[Transform](https://tumiz.github.io/scenario/Transform.html)
[Rotation](https://tumiz.github.io/scenario/Rotation.html)
[Color](https://tumiz.github.io/scenario/Color.html)

[Top](#Scenario-Tool-Kit)



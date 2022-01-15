*Copyright (c) Tumiz. Distributed under the terms of the GPL-3.0 License.*

py3d is a 3d computational geometry library.

It is designed to be simple, stable and customizable:

* simple means api will be less than usual and progressive
* stable means it will have less dependeces and modules, and it will be fully tested
* customizable means it will be a libaray rather than an application, it only provide data structures and functions handling basic geometry concepts

For more information, please visit [https://tumiz.github.io/scenario/]()

# Vector3 --Type for position, velocity & scale

**Vector3** represents point or position, velocity and scale. Note! Angular velocity cant be represented by this type, it should be represented by Rotation3 which will indroduced in next section. It is a class inheriting numpy.ndarray, so it is also ndarray.

## Defination

```python
Vector3(x:int|float|list|tuple|ndarray,y:int|float,z:int|float,n:int):Vector3
```

Vector3 can be a vector or a collection of vectors.

```python
from py3d import Vector3
from numpy import array
a=Vector3(1,2,3)
b=Vector3([1,2,3])
c=Vector3((1,2,3))
d=Vector3(array([1,2,3]))
e=Vector3(1,2,3,4)
a,b,c,d,e
```

```
(Vector3([1., 2., 3.]),
 Vector3([1., 2., 3.]),
 Vector3([1., 2., 3.]),
 Vector3([1., 2., 3.]),
 Vector3([[1., 2., 3.],
          [1., 2., 3.],
          [1., 2., 3.],
          [1., 2., 3.]]))
```

```python
Vector3.Rand(n:int):Vector3
```

Return a random vector or a collection of random vectors.

```python
Vector3.Zeros(n:int):Vector3
```

Return a zero vector or a collection of zero vectors.

```python
Vector3.Ones(n:int):Vector3
```

Return a vector or a collection of vectors filled with 1

```python
from py3d import Vector3
Vector3.Rand(4),Vector3.Zeros(4),Vector3.Ones(4)
```

```
(Vector3([[0.00240872, 0.06259652, 0.58789827],
          [0.84172269, 0.54447431, 0.02050995],
          [0.50090265, 0.00939204, 0.95925715],
          [0.72912007, 0.97297814, 0.65798418]]),
 Vector3([[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]]),
 Vector3([[1., 1., 1.],
          [1., 1., 1.],
          [1., 1., 1.],
          [1., 1., 1.]]))
```

```python
from py3d import Vector3
Vector3([1,2,3,4,5,6,7,8,9]),Vector3([[1,2,3],[4,5,6],[7,8,9]])
```

```
(Vector3([[1., 2., 3.],
          [4., 5., 6.],
          [7., 8., 9.]]),
 Vector3([[1., 2., 3.],
          [4., 5., 6.],
          [7., 8., 9.]]))
```

```python
from py3d import Vector3
Vector3(1,2,3,5),Vector3(y=1,n=4),Vector3(x=1,n=6)
```

```
(Vector3([[1., 2., 3.],
          [1., 2., 3.],
          [1., 2., 3.],
          [1., 2., 3.],
          [1., 2., 3.]]),
 Vector3([[0., 1., 0.],
          [0., 1., 0.],
          [0., 1., 0.],
          [0., 1., 0.]]),
 Vector3([[1., 0., 0.],
          [1., 0., 0.],
          [1., 0., 0.],
          [1., 0., 0.],
          [1., 0., 0.],
          [1., 0., 0.]]))
```

```python
from py3d import Vector3
from numpy import array, equal
a=Vector3(array([1,2,3]))
b=Vector3(a)
a==b,id(a),id(b)
```

```
(True, 140613749197056, 140613749196832)
```

## Deep copy

```python
.copy()
```

It will return deep copy of origin vector, and their value are equal.

```python
from py3d import Vector3
a=Vector3(1,2,3)
b=a
c=a.copy() # deep copy
id(a),id(b),id(c), a==c
```

```
(140613746450832, 140613746450832, 140613746451168, True)
```

```python
from py3d import Vector3
points=Vector3.Rand(5)
print(points.norm())
points_copy=points.copy()
points==points_copy
```

```
[[1.08873624]
 [0.56201636]
 [0.81603114]
 [0.69572861]
 [1.33044297]]





array([[ True],
       [ True],
       [ True],
       [ True],
       [ True]])
```

## Modify

```python
from py3d import Vector3
points=Vector3(1,2,3,4)
points
```

```
Vector3([[1., 2., 3.],
         [1., 2., 3.],
         [1., 2., 3.],
         [1., 2., 3.]])
```

```python
points[2]=Vector3(-1,-2,-3)
points
```

```
Vector3([[ 1.,  2.,  3.],
         [ 1.,  2.,  3.],
         [-1., -2., -3.],
         [ 1.,  2.,  3.]])
```

```python
points[0:2]=Vector3.Ones(2)
points
```

```
Vector3([[ 1.,  1.,  1.],
         [ 1.,  1.,  1.],
         [-1., -2., -3.],
         [ 1.,  2.,  3.]])
```

## Reverse

```python
.reverse():ndarray
```

```python
from py3d import *
a=Vector3.Rand(3)
print(a)
a.reverse()
print(a)
a.reversed()
```

```
[[0.37239685 0.85223555 0.27793704]
 [0.75213452 0.16901494 0.44511578]
 [0.9494015  0.35997485 0.57413589]]
[[0.9494015  0.35997485 0.57413589]
 [0.75213452 0.16901494 0.44511578]
 [0.37239685 0.85223555 0.27793704]]





Vector3([[0.37239685, 0.85223555, 0.27793704],
         [0.75213452, 0.16901494, 0.44511578],
         [0.9494015 , 0.35997485, 0.57413589]])
```

## Append

```python
.append(Vector3|ndarray):ndarray
```

```python
from py3d import *
a=Vector3.Rand(4)
a.append(Vector3(1,2,3,2))
a
```

```
Vector3([[0.1919075 , 0.46747677, 0.91061577],
         [0.02682452, 0.15863966, 0.5067785 ],
         [0.83158459, 0.27005634, 0.35526737],
         [0.65509237, 0.54353389, 0.11015612],
         [1.        , 2.        , 3.        ],
         [1.        , 2.        , 3.        ]])
```

## Insert

```python
from py3d import *
a=Vector3.Rand(4)
a.insert(2,Vector3(1,2,3,3))
a
```

```
Vector3([[0.6605133 , 0.37618622, 0.64276519],
         [0.38142681, 0.40017373, 0.13127457],
         [1.        , 2.        , 3.        ],
         [1.        , 2.        , 3.        ],
         [1.        , 2.        , 3.        ],
         [0.21344712, 0.00533367, 0.50443668],
         [0.21560269, 0.51254746, 0.65253392]])
```

```python
from py3d import *
a=Vector3.Rand(4)
a.insert(slice(0,3),Vector3(1,2,3))
a
```

```
Vector3([[1.        , 2.        , 3.        ],
         [0.79471519, 0.74496138, 0.68758799],
         [1.        , 2.        , 3.        ],
         [0.68778039, 0.18272503, 0.15025641],
         [1.        , 2.        , 3.        ],
         [0.78909031, 0.89734503, 0.50305253],
         [0.39830959, 0.40794724, 0.06154772]])
```

```python
from py3d import *
a=Vector3.Rand(4)
a.insert(0,Vector3(1,2,3))
a
```

```
Vector3([[1.        , 2.        , 3.        ],
         [0.36243349, 0.90058189, 0.91439372],
         [0.50756061, 0.16305892, 0.63210915],
         [0.07187428, 0.21402741, 0.43172284],
         [0.6327147 , 0.83150476, 0.40701695]])
```

## Remove

```python
from py3d import *
a=Vector3.Rand(4)
print(a)
a.remove(0)
a
```

```
[[0.53362679 0.9637612  0.79709125]
 [0.9183582  0.69815294 0.9979033 ]
 [0.97920985 0.57807659 0.72873601]
 [0.62200499 0.23591995 0.53537224]]





Vector3([[0.9183582 , 0.69815294, 0.9979033 ],
         [0.97920985, 0.57807659, 0.72873601],
         [0.62200499, 0.23591995, 0.53537224]])
```

```python
from py3d import *
a=Vector3.Rand(5)
print(a)
a.remove(slice(2,4))
a
```

```
[[0.61816345 0.21342644 0.06906031]
 [0.44855753 0.41317524 0.27265141]
 [0.981912   0.2943863  0.77828021]
 [0.4782964  0.40162783 0.28036749]
 [0.16483228 0.9366734  0.23671958]]





Vector3([[0.61816345, 0.21342644, 0.06906031],
         [0.44855753, 0.41317524, 0.27265141],
         [0.16483228, 0.9366734 , 0.23671958]])
```

```python
from py3d import *
a=Vector3.Rand(5)
print(a)
a.remove(slice(2,4))
a
```

```
[[0.23022599 0.79078078 0.83306751]
 [0.76219755 0.62387302 0.94054235]
 [0.38409679 0.91891268 0.21859557]
 [0.0472911  0.81482236 0.52050563]
 [0.55440996 0.23135002 0.03196446]]





Vector3([[0.23022599, 0.79078078, 0.83306751],
         [0.76219755, 0.62387302, 0.94054235],
         [0.55440996, 0.23135002, 0.03196446]])
```

## Discrete difference

```python
.diff(n:int):Vector3
```

```python
from py3d import Vector3
points=Vector3([
    [1,2,1],
    [2,3,1],
    [4,6,2],
    [8,3,0]
])
points.diff(),points.diff(2)
```

```
(Vector3([[ 1.,  1.,  0.],
          [ 2.,  3.,  1.],
          [ 4., -3., -2.]]),
 Vector3([[ 1.,  2.,  1.],
          [ 2., -6., -3.]]))
```

## Cumulative Sum

```python
.cumsum():Vector3
```

Return the cumulative sum of the elements along a given axis.

```python
from py3d import Vector3
points=Vector3([
    [1,2,1],
    [2,3,1],
    [4,6,2],
    [8,3,0]
])
points.cumsum()
```

```
Vector3([[ 1.,  2.,  1.],
         [ 3.,  5.,  2.],
         [ 7., 11.,  4.],
         [15., 14.,  4.]])
```

## Add

```python
from py3d import Vector3
Vector3(1,2,3)+Vector3(2,3,4)
```

```
Vector3([3., 5., 7.])
```

```python
from py3d import Vector3
Vector3.Zeros(3)+Vector3.Ones(3)
```

```
Vector3([[1., 1., 1.],
         [1., 1., 1.],
         [1., 1., 1.]])
```

```python
from py3d import Vector3
a=Vector3([1,2,3,4,5,6,7,8,9,-1,-2,-3])
b=Vector3([1,-2,-4,-5,-1,-4,3,5,6,9,10,8])
a+b
```

```
Vector3([[ 2.,  0., -1.],
         [-1.,  4.,  2.],
         [10., 13., 15.],
         [ 8.,  8.,  5.]])
```

## Subtract

```python
from py3d import Vector3
Vector3(1,2,3)-Vector3(-1,-2,-3)
```

```
Vector3([2., 4., 6.])
```

```python
from py3d import Vector3
Vector3([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])-Vector3(1,-1,3,5)
```

```
Vector3([[ 0.,  3.,  0.],
         [ 3.,  6.,  3.],
         [ 6.,  9.,  6.],
         [ 9., 12.,  9.],
         [12., 15., 12.]])
```

## Multiply

### Multiply a number

```python
from py3d import Vector3
a=Vector3(1,-2,3)*3
b=3*Vector3(1,-2,3)
a,b,a==b
```

```
(Vector3([ 3., -6.,  9.]), Vector3([ 3., -6.,  9.]), True)
```

### Multiply element by element

support multiplication between Vector3,Numpy.ndarray,list and tuple.

```python
from py3d import Vector3
from numpy import array
Vector3(1,-2,3)*Vector3(1,-1,3),\
Vector3(1,-2,3)*array([1,-1,3]),\
array([1,-1,3])*Vector3(1,-2,3),\
Vector3(1,-1,3)*[1,-2,3],\
(1,-1,3)*Vector3(1,-2,3)
```

```
(Vector3([1., 2., 9.]),
 Vector3([1., 2., 9.]),
 Vector3([1., 2., 9.]),
 Vector3([1., 2., 9.]),
 Vector3([1., 2., 9.]))
```

### Dot product

Two vectors' dot product can be used to calculate angle between them. If angle

$\bf{a}\cdot\bf{b}=|\bf{a}|\cdot|\bf{b}|\cdot cos\theta$

$\bf{a}\cdot\bf{b}=\bf{b}\cdot\bf{a}$

*.dot(Vector3):Vector3*

dot() will return a new Vector3, the original one wont be changed

```python
from py3d import Vector3
from numpy import cos
a=Vector3(1,-2,3)
b=Vector3(0,4,-1)
product=a.dot(b) # dot product
theta=a.angle_to_vector(b)
print(a.norm(),b.norm(),cos(theta))
print(a.norm()*b.norm()*cos(theta),product)
```

```
3.7416573867739413 4.123105625617661 -0.7130240959073809
-11.000000000000002 -11.0
```

```python
a.dot(b),b.dot(a),a.dot(b)==b.dot(a)
```

```
(-11.0, -11.0, True)
```

```python
from py3d import Vector3
a=Vector3.Rand(4)
b=Vector3.Rand(4)
a.dot(b)
```

```
Vector3([[0.52828999],
         [0.30207483],
         [1.10503466],
         [0.68242945]])
```

### Cross product

$ \bf{a}\times\bf{b}=|\bf{a}|\cdot|\bf{b}|\cdot sin\theta$

$\bf{a}\times\bf{b}=-\bf{b}\times\bf{a}$

```python
.cross(Vector3):Vector3
```

cross() will return a new Vector3, the original one wont be changed.

```python
from py3d import Vector3
a=Vector3(1,2,0)
b=Vector3(0,-1,3)
c=a.cross(b)
a.cross(b),b.cross(a) # cross product
```

```
(Vector3([ 6., -3., -1.]), Vector3([-6.,  3.,  1.]))
```

array([1,2,0]).cross(Vector3(0,-1,3)) is not allowed since numpy.ndarray has no such a function to do cross product. But you can do it by a global function numpy.cross(array1, array2) like this

```python
from numpy import cross,array
from py3d import Vector3
cross(array([1,2,0]), Vector3(0,-1,3))
```

```
array([ 6., -3., -1.])
```

Have a look to see the origin vectors and the product vector

```python
from py3d import Vector3
v1=Vector3(1,2,0)
v2=Vector3(0,-1,3)
vp=Vector3(1,2,0).cross(Vector3(0,-1,3))
```

```python
from py3d import Vector3
a=Vector3.Rand(4)
b=Vector3.Rand(4)
c=a.cross(b)
```

## Divide

### Divide by scalar

```python
from py3d import Vector3
Vector3(1,2,3)/3
```

```
Vector3([0.33333333, 0.66666667, 1.        ])
```

```python
from py3d import Vector3
a=Vector3(3,0,3)
b=a/3
a/=3
a,b
```

```
(Vector3([1., 0., 1.]), Vector3([1., 0., 1.]))
```

### Divide by vector

```python
from py3d import Vector3
Vector3(1,2,3)/Vector3(1,2,3)
```

```
Vector3([1., 1., 1.])
```

### Divide by Numpy.ndarray, list and tuple

Vector3 is divided element by element

```python
from py3d import Vector3
from numpy import array
Vector3(1,2,3)/array([1,2,3]), Vector3(1,2,3)/[1,2,3], Vector3(1,2,3)/(1,2,3)
```

```
(Vector3([1., 1., 1.]), Vector3([1., 1., 1.]), Vector3([1., 1., 1.]))
```

## Compare

```python
from py3d import *
a=Vector3(1,0,0.7)
b=Vector3(1.0,0.,0.7)
c=Vector3(1.1,0,0.7)
a==b,b==c,a!=c
```

```
(True, False, True)
```

```python
from py3d import Vector3
a=Vector3([[1,2,3],
           [4,5,6],
           [7,8,9]])
b=Vector3([[1,1,3],
           [4,5,6],
           [7,1,9]])
a==b
```

```
array([[False],
       [ True],
       [False]])
```

## Angle

```python
.angle_to_vector(v:Vector3):float|ndarray
```

It will return the angle (in radian) between two vector. The angle is always positive and smaller than $\pi$.

```python
from py3d import Vector3
v1=Vector3(1,-0.1,0)
v2=Vector3(0,1,0)
v1.angle_to_vector(v2),v2.angle_to_vector(v1)
```

```
(1.6704649792860586, 1.6704649792860586)
```

```python
from py3d import Vector3
a=Vector3([[1,2,3],
           [4,5,6],
           [7,8,9]])
b=Vector3([[1,1,3],
           [4,5,6],
           [7,1,9]])
a.angle_to_vector(b)
```

```
Vector3([[2.57665272e-01],
         [2.10734243e-08],
         [5.24348139e-01]])
```

```python
.angle_to_plane(normal:Vector3):float|ndarray
```

It will return the angle (in radian) between a vector and a plane. Result will be positive when normal and the vector have same direction, 0 when the plane and the vector is parallel, and negtive when normal and the vector have different direction.

```python
from py3d import Vector3
v=Vector3(1,-0.1,0)
normal=Vector3(0,1,0)
v.angle_to_plane(normal)
```

```
-0.09966865249116208
```

## Rotation

```python
.rotation_to(Vector3):Vector3,float
```

It will return axis-angle tuple representing the rotation from this vector to another

```python
from py3d import Vector3
v1=Vector3(1,-0.1,0)
v2=Vector3(0,1,0)
v1.rotation_to(v2),v2.rotation_to(v1)
```

```
((Vector3([-0.,  0.,  1.]), 1.6704649792860586),
 (Vector3([ 0.,  0., -1.]), 1.6704649792860586))
```

```python
from py3d import Vector3
a=Vector3([[1,-0.1,0],
        [0,1,0]])
b=Vector3([[0,1,0],
          [1,-0.1,0]])
a.rotation_to(b)
```

```
(Vector3([[-0.,  0.,  1.],
          [ 0.,  0., -1.]]),
 Vector3([[1.67046498],
          [1.67046498]]))
```

## Perpendicular

$\bf{a}\perp\bf{b}\Leftrightarrow\bf{a}\cdot\bf{b}=0$

$\bf{a}\perp\bf{b}\Leftrightarrow<\bf{a},\bf{b}>=\pi/2$

```python
    .is_perpendicular_to_vector(v:Vector3): bool
    .is_perpendicular_to_plane(normal:Vector3): bool
```

```python
from py3d import Vector3
a=Vector3(0,1,1)
b=Vector3(1,0,0)
a.is_perpendicular_to_vector(b), a.angle_to_vector(b)
```

```
(True, 1.5707963267948966)
```

## Parallel

$\bf{a}//\bf{b}(\bf{b}\ne\bf{0})\Leftrightarrow\bf{a}=\lambda\bf{b}$

```python
from py3d import Vector3
a=Vector3(1,2,3)
b=Vector3(2,4,6)
plane = Vector3(1,2,)
a.is_parallel_to_vector(b),a==b
```

```
(True, False)
```

$\bf{v}\perp\bf{0}, \bf{v}\cdot\bf{0}=0$ is always true no matter what $\bf{v}$ is

```python
from py3d import Vector3
a=Vector3(1,2,3)
b=Vector3(-2,3,9)
a.dot(Vector3()),a.is_parallel_to_vector(Vector3()),b.is_parallel_to_vector(b)
```

```
/mnt/d/codes/scenario/py3d/py3d/vector3.py:52: RuntimeWarning: invalid value encountered in true_divide
  return self/l





(0.0, False, True)
```

## Projection

```python
.scalar_projection(v:Vector3):float
```

```python
.vector_projection(v:Vector3):Vector3
```

```python
from py3d import Vector3
a=Vector3(2,1,1)
b=Vector3(1,0,0)
a.scalar_projection(b),a.vector_projection(b)
```

```
(2.0, Vector3([2., 0., 0.]))
```

```python
from py3d import Vector3
a=Vector3(1,2,3)
p0=Vector3()
p1=Vector3(1,0,0)
a.projection_point_on_line(p0,p1)
```

```
Vector3([1., 0., 0.])
```

## Area

```python
.area(Vector3):float
```

It will return area of triangle constucted by two vectors.

```python
.area(Vector3,Vector3):float
```

It will return area of triangle constructed by three points.

```python
from py3d import Vector3
triangle=Vector3([[1,2,3],
                [1,0,0],
                [0,1,0]])
triangle.area()
```

```
2.345207879911715
```

## Distance, Length, Norm

```python
.norm():float
```

```python
from py3d import Vector3
Vector3(1,2,3).norm()
```

```
3.7416573867739413
```

You can use this function to calculate distance between two points.

```python
point1=Vector3(1,2,3)
point2=Vector3(-10,87,11)
distance=(point1-point2).norm()
print(distance)
```

```
86.08135686662938
```

```python
from py3d import Vector3
points=Vector3.Rand(5)
points.norm()
```

```
array([[1.00952545],
       [0.48242001],
       [0.89271163],
       [0.89204501],
       [0.73384055]])
```

Calculate distances between a point and a collection of points

```python
from py3d import Vector3
p=Vector3(1,-1,0)
points=Vector3.Rand(7)
points,(p-points).norm()
```

```
(Vector3([[0.92626725, 0.11644017, 0.08594409],
          [0.16715586, 0.38188221, 0.67990894],
          [0.81794796, 0.89077802, 0.00486246],
          [0.45000451, 0.48546387, 0.55692739],
          [0.68920177, 0.88953077, 0.46585984],
          [0.7319839 , 0.25516827, 0.40738851],
          [0.93042351, 0.79431006, 0.1907029 ]]),
 array([[1.12216824],
        [1.75085807],
        [1.89952839],
        [1.67906702],
        [1.97077332],
        [1.34656801],
        [1.80575665]]))
```

## Normalize

`<font color="red">`*! Zero vector can not be normalized* `</font>`

**normalized()**, get a new vector, which is the unit vector of the origin

```python
from py3d import Vector3
v=Vector3(1,2,3)
v.normalized()
```

```
Vector3([0.26726124, 0.53452248, 0.80178373])
```

```python

```

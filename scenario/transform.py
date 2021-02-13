from typing import Optional, Tuple, Union

from numpy.core.defchararray import equal

from scenario.server import Source
from numpy import *
from math import acos, sin, cos, cos, atan2, sqrt, pi


class Vector3(ndarray):
    def __new__(cls, x=0, y=0, z=0, n=1):
        if isinstance(x, list) or isinstance(x, tuple) or isinstance(x, ndarray):
            data = array(x, dtype=float)
            if data.size % 3:
                raise("input data size must be multiple of 3")
            else:
                n = data.size//3
                if n == 0:
                    raise("input data is empty")
                elif n == 1:
                    data = data.reshape(3)
                else:
                    data = data.reshape(n, 3)
        elif n > 1:
            data = tile(array([x, y, z], dtype=float), (n, 1))
        else:
            data = array([x, y, z], dtype=float)
        obj = ndarray.__new__(
            cls, data.shape, buffer=data)
        return obj

    @classmethod
    def Rand(cls, n: int):
        return Vector3(random.rand(3, n))

    @classmethod
    def Zeros(cls, n: int):
        return Vector3(zeros((n, 3)))

    @classmethod
    def Ones(cls, n: int):
        return Vector3(ones((n, 3)))

    def norm(self) -> ndarray:  # norm
        if self.ndim > 1:
            return linalg.norm(self, axis=1, keepdims=True)
        else:
            return linalg.norm(self)

    def normalize(self) -> None:
        l = self.norm()
        try:
            self /= l
        except:
            ValueError("zero vector can not be normalized")

    def normalized(self) -> Optional[ndarray]:  # unit vector, direction vector
        l = self.norm()
        try:
            return self/l
        except:
            return None

    def reverse(self) -> None:
        self[:] = flipud(self)

    def reversed(self) -> ndarray:
        return flipud(self)

    def append(self, v) -> ndarray:
        self[:]=concatenate((self,v), axis=0)

    def dot(self, v: ndarray) -> ndarray:
        if v.ndim > 1:
            d = dot(self, v.T).diagonal()
            return array(d.reshape(d.size, 1))
        else:
            return super().dot(v)

    def cross(self, v: ndarray) -> ndarray:
        return Vector3(cross(self, v))

    def angle_to_vector(self, to: ndarray) -> ndarray:
        cos = self.dot(to)/self.norm()/to.norm()
        return arccos(cos)

    def angle_to_plane(self, normal: ndarray) -> float:
        return pi/2 - self.angle_to_vector(normal)

    def rotation_to(self, to: ndarray) -> Tuple[ndarray, float]:
        axis = self.cross(to)
        angle = self.angle_to_vector(to)
        return axis, angle

    def is_parallel_to_vector(self, v: ndarray) -> bool:
        return self.normalized() == v.normalized()

    def is_parallel_to_plane(self, normal: ndarray) -> bool:
        return self.is_perpendicular_to_vector(normal)

    def is_perpendicular_to_vector(self, v: ndarray) -> bool:
        return self.dot(v) == 0

    def is_perpendicular_to_plane(self, normal: ndarray) -> bool:
        return self.is_parallel_to_vector(normal)

    def scalar_projection(self, v: ndarray) -> float:
        return self.dot(v).item()/v.norm()

    def vector_projection(self, v: ndarray) -> ndarray:
        return self.scalar_projection(v)/v.norm()*v

    def distance_to_line(self, p0: ndarray, p1: ndarray) -> float:
        v0 = p1-p0
        v1 = self-p0
        return v0.cross(v1).norm()/v0.norm()

    def distance_to_plane(self, n: ndarray, p: ndarray) -> float:
        v = self - p
        return v.scalar_projection(n)

    def projection_point_on_line(self, p0: ndarray, p1: ndarray) -> ndarray:
        return p0+(self-p0).vector_projection(p1-p0)

    def projection_point_on_plane(self, normal: ndarray, point: ndarray) -> ndarray:
        return self+(point-self).vector_projection(normal)

    def area(self) -> float:
        if self.ndim > 1 and self.shape[0] == 3:
            v0 = self[1]-self[0]
            v1 = self[2]-self[0]
            return v0.cross(v1).norm()/2
        else:
            raise "size should be (3,3)"

    def __eq__(self, v: ndarray):
        if isinstance(v, ndarray):
            if self.ndim == v.ndim and self.size == v.size:
                if self.ndim > 1:
                    return array(equal(self, v).all(axis=1, keepdims=True))
                else:
                    return equal(self, v).all().item()
            else:
                return False
        else:
            return array(equal(self, v))

    def __ne__(self, v: ndarray) -> bool:
        return not self.__eq__(v)

    def numpy(self):
        return array(self)

    def render(self, serial=True):
        s = Source("default")
        s.send_msg({
            "class": self.__class__.__name__,
            "data": {
                "data": self.tolist() if self.ndim > 1 else [self.tolist()],
                "serial": serial
            }
        })


class Rotation3(ndarray):
    def __new__(cls, matrix=eye(3)):
        return ndarray.__new__(cls, (3, 3), buffer=array(matrix, dtype=float))

    # rotate around body frame's axis
    @classmethod
    def IntrinsicEular(cls, x=0, y=0, z=0):
        return cls.Rx(x).dot(cls.Ry(y)).dot(cls.Rz(z))

    # rotate around parent frame's axis
    @classmethod
    def ExtrinsicEular(cls, x=0, y=0, z=0):
        return cls.Rz(z).dot(cls.Ry(y)).dot(cls.Rx(x))

    @classmethod
    def Quaternion(cls, x, y, z, w):
        return cls([
            [2*(pow(x, 2)+pow(w, 2))-1, 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 2*(pow(w, 2)+pow(y, 2))-1, 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 2*(pow(w, 2)+pow(z, 2))-1]
        ])

    @classmethod
    def Axis_angle(cls, axis, angle):
        axis_n = axis.norm()
        if axis_n:
            axis = axis/axis_n
            w = cos(angle/2)
            x, y, z = sin(angle/2)*axis
            return cls.Quaternion(x, y, z, w)
        else:
            return cls()

    @classmethod
    def Direction_change(cls, before, after):
        axis = before.cross(after)
        angle = acos(before.dot(after)/before.norm()/after.norm())
        return cls.Axis_angle(axis, angle)

    @classmethod
    def Rx(cls, a):
        return cls([
            [1, 0, 0],
            [0, cos(a), -sin(a)],
            [0, sin(a), cos(a)]
        ])

    @classmethod
    def Ry(cls, a):
        return cls([
            [cos(a), 0, sin(a)],
            [0, 1, 0],
            [-sin(a), 0, cos(a)]
        ])

    @classmethod
    def Rz(cls, a):
        return cls([
            [cos(a), -sin(a), 0],
            [sin(a), cos(a), 0],
            [0, 0, 1]
        ])

    def to_extrinsic_eular(self):
        x = atan2(self[2, 1], self[2, 2])
        y = atan2(-self[2, 0], sqrt(self[2, 1]**2+self[2, 2]**2))
        z = atan2(self[1, 0], self[0, 0])
        return [x, y, z]

    def to_instrinsic_eular(self):
        x = atan2(-self[1, 2], self[2, 2])
        y = atan2(self[0, 2], sqrt(self[1, 2]**2+self[2, 2]**2))
        z = atan2(-self[0, 1], self[0, 0])
        return [x, y, z]

    def to_quaternion(self):
        w = 0.5*sqrt(self[0, 0]+self[1, 1]+self[2, 2]+1)
        x = 0.5*sign(self[2, 1]-self[1, 2]) * \
            sqrt(max(0, self[0, 0]-self[1, 1]-self[2, 2]+1))
        y = 0.5*sign(self[0, 2]-self[2, 0]) * \
            sqrt(max(0, self[1, 1]-self[2, 2]-self[0, 0]+1))
        z = 0.5*sign(self[1, 0]-self[0, 1]) * \
            sqrt(max(0, self[2, 2]-self[0, 0]-self[1, 1]+1))
        return [x, y, z, w]

    def to_axis_angle(self):
        angle = acos((self[0, 0]+self[1, 1]+self[2, 2]-1)/2)
        axis = Vector3(
            self[2, 1]-self[1, 2],
            self[0, 2]-self[2, 0],
            self[1, 0]-self[0, 1]
        ).normalized()
        if axis is None:
            raise VelueError("axis is a zero vector")
        else:
            return axis, angle

    def rotate_x(self, angle):
        self.real = Rotation3.Rx(angle).dot(self)
        return self

    def rotate_y(self, angle):
        self.real = Rotation3.Ry(angle).dot(self)
        return self

    def rotate_z(self, angle):
        self.real = Rotation3.Rz(angle).dot(self)
        return self

    def rotate_axis(self, axis, angle):
        self.real = Rotation3.Axis_angle(axis, angle).dot(self)
        return self

    def __mul__(self, v):
        t = type(v)
        if t is float or t is int:
            axis, angle = self.to_axis_angle()
            angle *= v
            return Rotation3.Axis_angle(axis, angle)
        elif t is Rotation3:
            return self.dot(v)
        elif t is Vector3:
            return Vector3(self.dot(v))
        else:
            return None

    def __imul__(self, v):
        self = self*v
        return self

    def __eq__(self, v):
        return self.data == v.data

    def __ne__(self, v):
        return self.data != v.data

    @property
    def I(self):
        return linalg.inv(self)


class Transform:
    def __init__(self):
        self.position = Vector3()
        self.rotation = Rotation3()
        self.scale = Vector3(1, 1, 1)
        self.direction = Vector3(1, 0, 0)
        self.parent = None
        self.children = set()

    def add(self, *objs):
        for obj in objs:
            obj.parent = self
            self.children.add(obj)

    def world_position(self):
        parent = self.parent
        ret = self.position
        while parent:
            ret = parent*ret
            parent = parent.parent
        return ret

    def world_rotation(self):
        parent = self.parent
        ret = self.rotation
        while parent:
            ret = parent.rotation*ret
            parent = parent.parent
        return ret

    def __mul__(self, v):
        return self.rotation*(self.scale*v)+self.position

    def lookat(self, destination):
        self.rotation = Rotation3.Direction_change(
            self.direction, destination-self.position)

    def info(self):
        return {"position": self.world_position().tolist(), "rotation": self.world_rotation().to_extrinsic_eular(), "scale": self.scale.tolist()}

from typing import Optional, Tuple, Union
from numpy.core.defchararray import equal

from scenario.server import Source
import numpy
import math

class Vector3(numpy.ndarray):
    as_points = 0
    as_connected_points = 1
    as_vectors = 2
    as_connected_vectors = 3
    as_vectors_from_given_points = 4
    def __new__(cls, x=0, y=0, z=0, n=1):
        if isinstance(x, list) or isinstance(x, tuple) or isinstance(x, numpy.ndarray):
            data = numpy.array(x, dtype=float)
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
            data = numpy.tile(numpy.array([x, y, z], dtype=float), (n, 1))
        else:
            data = numpy.array([x, y, z], dtype=float)
        return data.view(cls).copy()

    @classmethod
    def Rand(cls, n: int):
        return Vector3(numpy.random.rand(3, n))

    @classmethod
    def Zeros(cls, n: int):
        return Vector3(numpy.zeros((n, 3)))

    @classmethod
    def Ones(cls, n: int):
        return Vector3(numpy.ones((n, 3)))

    def norm(self) -> numpy.ndarray:  # norm
        if self.ndim > 1:
            return numpy.linalg.norm(self, axis=1, keepdims=True)
        else:
            return numpy.linalg.norm(self)

    def normalize(self) -> numpy.ndarray:
        l = self.norm()
        try:
            self /= l
        except:
            ValueError("zero vector can not be normalized")
        return self

    def normalized(self) -> Optional[numpy.ndarray]:  # unit vector, direction vector
        l = self.norm()
        try:
            return self/l
        except:
            return None

    def reverse(self) -> numpy.ndarray:
        self[:] = numpy.flipud(self)
        return self

    def reversed(self) -> numpy.ndarray:
        return numpy.flipud(self)

    def append(self, v) -> numpy.ndarray:
        end0 = self.size//3
        end1 = (self.size+v.size)//3
        self.resize(end1, 3, refcheck=False)
        self[end0:end1, :] = v
        return self

    def insert(self, pos, v) -> numpy.ndarray:
        new = numpy.insert(self, pos, v, axis=0)
        self.resize(new.size//3, 3, refcheck=False)
        self[:] = new
        return self

    def remove(self, pos) -> numpy.ndarray:
        new = numpy.delete(self, pos, axis=0)
        self.resize(new.size//3, 3, refcheck=False)
        self[:] = new
        return self

    def diff(self, n = 1) -> numpy.ndarray:
        if self.ndim > 1:
            return numpy.diff(self, n, axis=0)
        else:
            raise "Only vector list has discrete difference"

    def cumsum(self) -> numpy.ndarray:
        if self.ndim > 1:
            return super().cumsum(axis=0)
        else:
            return self

    def dot(self, v: numpy.ndarray) -> numpy.ndarray:
        if v.ndim > 1:
            d = numpy.dot(self, v.T).diagonal()
            return numpy.array(d.reshape(d.size, 1))
        else:
            return super().dot(v)

    def cross(self, v: numpy.ndarray) -> numpy.ndarray:
        return Vector3(numpy.cross(self, v))

    def angle_to_vector(self, to: numpy.ndarray) -> numpy.ndarray:
        cos = self.dot(to)/self.norm()/to.norm()
        return math.acos(cos)

    def angle_to_plane(self, normal: numpy.ndarray) -> float:
        return math.pi/2 - self.angle_to_vector(normal)

    def rotation_to(self, to: numpy.ndarray) -> Tuple[numpy.ndarray, float]:
        axis = self.cross(to)
        angle = self.angle_to_vector(to)
        return axis, angle

    def is_parallel_to_vector(self, v: numpy.ndarray) -> bool:
        return self.normalized() == v.normalized()

    def is_parallel_to_plane(self, normal: numpy.ndarray) -> bool:
        return self.is_perpendicular_to_vector(normal)

    def is_perpendicular_to_vector(self, v: numpy.ndarray) -> bool:
        return self.dot(v) == 0

    def is_perpendicular_to_plane(self, normal: numpy.ndarray) -> bool:
        return self.is_parallel_to_vector(normal)

    def scalar_projection(self, v: numpy.ndarray) -> float:
        return self.dot(v).item()/v.norm()

    def vector_projection(self, v: numpy.ndarray) -> numpy.ndarray:
        return self.scalar_projection(v)/v.norm()*v

    def distance_to_line(self, p0: numpy.ndarray, p1: numpy.ndarray) -> float:
        v0 = p1-p0
        v1 = self-p0
        return v0.cross(v1).norm()/v0.norm()

    def distance_to_plane(self, n: numpy.ndarray, p: numpy.ndarray) -> float:
        v = self - p
        return v.scalar_projection(n)

    def projection_point_on_line(self, p0: numpy.ndarray, p1: numpy.ndarray) -> numpy.ndarray:
        return p0+(self-p0).vector_projection(p1-p0)

    def projection_point_on_plane(self, normal: numpy.ndarray, point: numpy.ndarray) -> numpy.ndarray:
        return self+(point-self).vector_projection(normal)

    def area(self) -> float:
        if self.ndim > 1 and self.shape[0] == 3:
            v0 = self[1]-self[0]
            v1 = self[2]-self[0]
            return v0.cross(v1).norm()/2
        else:
            raise "size should be (3,3)"

    def __eq__(self, v: numpy.ndarray):
        if isinstance(v, numpy.ndarray):
            if self.ndim == v.ndim and self.size == v.size:
                if self.ndim > 1:
                    return array(equal(self, v).all(axis=1, keepdims=True))
                else:
                    return equal(self, v).all().item()
            else:
                return False
        else:
            return array(equal(self, v))

    def __ne__(self, v: numpy.ndarray) -> bool:
        return not self.__eq__(v)

    def numpy(self):
        return array(self)

    def render(self, color="white", mode=0, size = 1, start_points=None):
        if mode == Vector3.as_points:
            sp = None
            ep = self
        elif mode == Vector3.as_connected_points:
            if self.ndim > 1:
                sp = self[0:-1]
                ep = self[1::]
            else:
                raise "Single vector can not be connected"
        elif mode == Vector3.as_vectors:
            sp = Vector3()
            ep = self
        elif mode == Vector3.as_connected_vectors:
            ep = self.cumsum()
            sp = insert(ep[0:-1], 0, Vector3(), axis=0)
        elif mode == Vector3.as_vectors_from_given_points:
            sp = start_points
            ep = start_points + self
        if sp is None:
            sp = []
        elif sp.ndim > 1:
            sp = sp.tolist()
        else:
            sp = [sp.tolist()]
        if ep is None:
            ep = []
        elif ep.ndim > 1:
            ep = ep.tolist()
        else:
            ep = [ep.tolist()]
        s = Source("default")
        s.send_msg(Source.action_draw,{
            "class": self.__class__.__name__,
            "data": {
                "color": color,
                "size": size,
                "start_points": sp,
                "end_points": ep,
            }
        })

 
class Rotation3(numpy.ndarray):
    def __new__(cls, matrix=numpy.eye(3)):
        return numpy.ndarray.__new__(cls, (3, 3), buffer=array(matrix, dtype=float))

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

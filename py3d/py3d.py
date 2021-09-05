from typing import Optional, Tuple
import numpy
from toweb import Space

class Vector3(numpy.ndarray):
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
    def Rand(cls, n = 1):
        return Vector3(numpy.random.rand(3, n))

    # construct a vector3 list of length n and filled with zero
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

    # unit vector, direction vector
    def normalized(self) -> Optional[numpy.ndarray]:
        l = self.norm()
        try:
            return self/l
        except:
            return None

    # [1,2,3] => [3,2,1]
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

    def diff(self, n=1) -> numpy.ndarray:
        if self.ndim > 1:
            return numpy.diff(self, n, axis=0)
        else:
            raise "Only vector list has discrete difference"

    def cumsum(self) -> numpy.ndarray:
        if self.ndim > 1:
            return super().cumsum(axis=0)
        else:
            return self

    def dot(self,v) -> numpy.ndarray:
        if type(v) is Vector3:
            return (self*v).sum(axis=1)
        else:
            return numpy.dot(self,v)

    def cross(self, v: numpy.ndarray) -> numpy.ndarray:
        return Vector3(numpy.cross(self, v))

    def angle_to_vector(self, to: numpy.ndarray) -> numpy.ndarray:
        cos = self.dot(to)/self.norm()/to.norm()
        return numpy.arccos(cos)

    def angle_to_plane(self, normal: numpy.ndarray) -> float:
        return numpy.pi/2 - self.angle_to_vector(normal)

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
        # n: normal vector of the plane
        # p: a point on the plane
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
                    return numpy.array(numpy.equal(self, v).all(axis=1, keepdims=True))
                else:
                    return numpy.equal(self, v).all().item()
            else:
                return False
        else:
            return numpy.array(numpy.equal(self, v))

    def __ne__(self, v: numpy.ndarray) -> bool:
        return not self.__eq__(v)

    def numpy(self):
        return numpy.array(self)

    def render_as_points(self, page=""):
        p=Space(page)
        p.render_points(self.flatten().tolist())

    def render_as_connected_vectors(self, page=""):
        start_points=Vector3().append(self[0:-1])
        p=Space(page)
        p.render_arrows(start_points.tolist(),self.tolist())

    def render_as_origin_vectors(self, page=""):
        start_points=Vector3.Zeros(len(self))
        p=Space(page)
        p.render_arrows(start_points.tolist(),self.tolist())

class Rotation3(numpy.ndarray):
    def __new__(cls, matrix=numpy.eye(3)):
        return numpy.ndarray.__new__(cls, (3, 3), buffer=numpy.array(matrix, dtype=float))

    # rotate around body frame's axis
    @classmethod
    def EularIntrinsic(cls, x=0, y=0, z=0):
        return cls.Rz(x).T.dot(cls.Ry(y).T).dot(cls.Rx(z).T)

    # rotate around parent frame's axis
    @classmethod
    def EularExtrinsic(cls, x=0, y=0, z=0):
        return cls.Rx(z).dot(cls.Ry(y)).dot(cls.Rz(x))

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
            w = numpy.cos(angle/2)
            x, y, z = numpy.sin(angle/2)*axis
            return cls.Quaternion(x, y, z, w)
        else:
            return cls()

    @classmethod
    def Direction_change(cls, before, after):
        axis = before.cross(after)
        angle = numpy.acos(before.dot(after)/before.norm()/after.norm())
        return cls.Axis_angle(axis, angle)

    @classmethod
    def Rx(cls, a):
        return cls([
            [1, 0, 0],
            [0, numpy.cos(a), numpy.sin(a)],
            [0, -numpy.sin(a), numpy.cos(a)]
        ])

    @classmethod
    def Ry(cls, a):
        return cls([
            [numpy.cos(a), 0, -numpy.sin(a)],
            [0, 1, 0],
            [numpy.sin(a), 0, numpy.cos(a)]
        ])

    @classmethod
    def Rz(cls, a):
        return cls([
            [numpy.cos(a), numpy.sin(a), 0],
            [-numpy.sin(a), numpy.cos(a), 0],
            [0, 0, 1]
        ])

    def to_extrinsic_eular(self):
        x = numpy.arctan2(self[1, 2], self[2, 2])
        y = numpy.arctan2(-self[0, 2], numpy.sqrt(self[1, 2]**2+self[2, 2]**2))
        z = numpy.arctan2(self[0, 1], self[0, 0])
        return [x, y, z]

    def to_instrinsic_eular(self):
        x = numpy.arctan2(-self[1, 2], self[2, 2])
        y = numpy.arctan2(self[0, 2], numpy.sqrt(self[1, 2]**2+self[2, 2]**2))
        z = numpy.arctan2(-self[0, 1], self[0, 0])
        return [x, y, z]

    def to_quaternion(self):
        w = 0.5*numpy.sqrt(self[0, 0]+self[1, 1]+self[2, 2]+1)
        x = 0.5*numpy.sign(self[2, 1]-self[1, 2]) * \
            numpy.sqrt(max(0, self[0, 0]-self[1, 1]-self[2, 2]+1))
        y = 0.5*sign(self[0, 2]-self[2, 0]) * \
            numpy.sqrt(max(0, self[1, 1]-self[2, 2]-self[0, 0]+1))
        z = 0.5*sign(self[1, 0]-self[0, 1]) * \
            numpy.sqrt(max(0, self[2, 2]-self[0, 0]-self[1, 1]+1))
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
        return numpy.linalg.inv(self)

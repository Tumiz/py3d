# Copyright (c) Tumiz.
# Distributed under the terms of the GPL-3.0 License.
import collections
import numpy
from .server import Space

pi = numpy.arccos(-1)

def sorted_sizes(*sizes):
    return tuple(list(dict(sorted(collections.Counter(sizes).items(), key=lambda v: v[1])).keys()))


def merge_shapes(*shapes):
    ret = ()
    ndim = max([len(s) for s in shapes])
    for i in range(ndim):
        sizes = []
        for s in shapes:
            if len(s) == 1 and s[-1] == 1:
                continue
            if len(s) > i:
                sizes.append(s[-1-i])
        ret = sorted_sizes(*sizes) + ret
    return ret


def force_assgin(v1, v2):
    try:
        numpy.copyto(v1, v2)
    except:
        if v1.size == v2.size:
            v1[:] = v2.reshape(v1.shape)
        else:
            v1[:] = v2[:, numpy.newaxis]


class Data(numpy.ndarray):
    # usually, d1 is the number of entities, and d3 is the size of one element.
    def __new__(cls, *shape):
        return numpy.empty(shape).view(cls).copy()

    def __len__(self):
        if self.ndim == 1:
            return 1
        return super().__len__()

    @property
    def n(self):
        return self.shape[:-1] if self.ndim > 1 else (1,)

    @classmethod
    def load(cls, path):
        return numpy.load(path).view(cls)

    def save(self, path):
        numpy.save(path, self)

    @classmethod
    def load_csv(cls, path):
        return numpy.loadtxt(path, delimiter=',').view(cls)

    def save_csv(self, path):
        numpy.savetxt(path, self, delimiter=',')


class Vector3(Data):
    def __new__(cls, x=0, y=0, z=0, n=()):
        x_ = numpy.array(x)
        if x_.ndim > 1 and x_.shape[-1] == 3:
            return x_.view(cls)
        y_ = numpy.array(y)
        z_ = numpy.array(z)
        shape = merge_shapes(x_.shape, y_.shape, z_.shape)
        n = (n,) if type(n) is int else n
        shape = *n, *shape, 3
        ret = super().__new__(cls, *shape)
        ret.x = x_
        ret.y = y_
        ret.z = z_
        return ret

    @classmethod
    def from_array(cls, v):
        array = numpy.array(v)
        assert array.ndim == 1 or array.ndim == 2
        if array.ndim == 1:
            assert array.size % 3 == 0
            tmp = array.reshape(array.size // 3, 3)
            return tmp.view(cls)
        else:
            assert array.shape[1] == 3
            return array.view(cls)

    @classmethod
    def Rand(cls, *n):
        return numpy.random.rand(*n, 3).view(cls)

    @classmethod
    def Rectangle(cls):
        return numpy.array([
            [1, 1, 0],
            [-1, 1, 0],
            [-1, -1, 0],
            [1, -1, 0]]).view(cls)

    @classmethod
    def RectangleIndexed(cls):
        return cls.Rectangle()[[0, 1, 2, 2, 3, 0]]

    @property
    def x(self):
        return self[..., 0].view(numpy.ndarray)

    @x.setter
    def x(self, v):
        force_assgin(self[..., 0], v)

    @property
    def y(self):
        return self[..., 1].view(numpy.ndarray)

    @y.setter
    def y(self, v):
        force_assgin(self[..., 1], v)

    @property
    def z(self):
        return self[..., 2].view(numpy.ndarray)

    @z.setter
    def z(self, v):
        force_assgin(self[..., 2], v)

    @property
    def H(self):
        # Homogeneous vector, a 3D vector has 4 numbers
        return numpy.insert(self, 3, 1, axis=self.ndim-1)

    @property
    def M(self):
        # mean vector
        return super().mean(axis=self.ndim-2)

    @property
    def U(self):
        # unit vector, direction vector
        n = self.norm()
        return numpy.divide(self, n, where=n != 0)

    def norm(self) -> numpy.ndarray:  # norm
        return numpy.linalg.norm(self, axis=self.ndim - 1, keepdims=True)

    def SST(self):
        return ((self-self.M).norm()**2).sum()

    def append(self, v) -> numpy.ndarray:
        return numpy.concatenate((self, v), axis=0).view(self.__class__)

    def insert(self, pos, v) -> numpy.ndarray:
        # wouldnt change self
        return numpy.insert(self, pos, v, axis=0)

    def remove(self, pos) -> numpy.ndarray:
        # wouldnt change self
        return numpy.delete(self, pos, axis=0)

    def diff(self, n=1) -> numpy.ndarray:
        return numpy.diff(self, n, axis=0)

    def cumsum(self) -> numpy.ndarray:
        return super().cumsum(axis=self.ndim-2)

    def mq(self, q) -> numpy.ndarray:
        # multiply quaternion
        p = Quaternion(0, self)
        return q.mq(p, byrow=False).mq(q.I).xyz

    def mt(self, v) -> numpy.ndarray:
        # multiply transform
        return (self.H @ v)[..., 0:3].view(self.__class__)

    def dot(self, v) -> numpy.ndarray:
        if type(v) is Vector3:
            product = self * v
            return product.sum(axis=product.ndim - 1, keepdims=True).view(numpy.ndarray)
        else:
            return numpy.dot(self, v)

    def cross(self, v: numpy.ndarray) -> numpy.ndarray:
        return numpy.cross(self, v).view(self.__class__)

    def angle_to_vector(self, to: numpy.ndarray) -> float:
        cos = self.dot(to) / self.norm() / to.norm()
        return numpy.arccos(cos)

    def angle_to_plane(self, normal: numpy.ndarray) -> float:
        return numpy.pi / 2 - self.angle_to_vector(normal)

    def is_parallel_to_vector(self, v: numpy.ndarray) -> bool:
        return self.U == v.U

    def is_parallel_to_plane(self, normal: numpy.ndarray) -> bool:
        return self.is_perpendicular_to_vector(normal)

    def is_perpendicular_to_vector(self, v: numpy.ndarray) -> bool:
        return self.dot(v) == 0

    def is_perpendicular_to_plane(self, normal: numpy.ndarray) -> bool:
        return self.is_parallel_to_vector(normal)

    def scalar_projection(self, v: numpy.ndarray) -> float:
        return self.dot(v) / v.norm()

    def vector_projection(self, v: numpy.ndarray) -> numpy.ndarray:
        return self.scalar_projection(v) * v / v.norm()

    def distance_to_line(self, p0: numpy.ndarray, p1: numpy.ndarray) -> float:
        v0 = p1 - p0
        v1 = self - p0
        return v0.cross(v1).norm() / v0.norm()

    def distance_to_plane(self, n: numpy.ndarray, p: numpy.ndarray) -> float:
        # n: normal vector of the plane
        # p: a point on the plane
        v = self - p
        return v.scalar_projection(n)

    def projection_on_line(
        self, p0: numpy.ndarray, p1: numpy.ndarray
    ) -> numpy.ndarray:
        return p0 + (self - p0).vector_projection(p1 - p0)

    def projection_on_plane(self, plane) -> numpy.ndarray:
        return self + (plane.position[:, numpy.newaxis] - self).vector_projection(
            plane.normal[:, numpy.newaxis]
        )

    def interp(self, xp, x):
        i=numpy.searchsorted(xp,x)
        x0=xp[i-1]
        x1=xp[i]
        d=(x-x0)/x1-x0
        f0=self[i-1]
        f1=self[i]
        return d*f0+(1-d)*f1

    def as_scaling(self) -> numpy.ndarray:
        ret = Transform(*self.n)
        ret[..., 0, 0] = self[..., 0]
        ret[..., 1, 1] = self[..., 1]
        ret[..., 2, 2] = self[..., 2]
        return ret

    def as_translation(self):
        ret = Transform(*self.n)
        ret[..., 3, 0] = self[..., 0]
        ret[..., 3, 1] = self[..., 1]
        ret[..., 3, 2] = self[..., 2]
        return ret

    def as_point(self, color=None):
        entity = Point(*self.n)
        entity.vertice = self
        if color is not None:
            entity.color = color
        return entity

    def as_vector(self, start=0):
        entity = Arrow(*self.n)
        entity.start = start
        entity.end = self
        return entity

    def as_line(self):
        vertice = numpy.repeat(self, 2, axis=self.ndim -
                               2)[..., 1:-1, :]
        entity = LineSegment(*vertice.n)
        entity.vertice = vertice
        return entity

    def as_linesegment(self):
        entity = LineSegment(*self.n)
        entity.vertice = self
        return entity

    def as_mesh(self):
        entity = Mesh.from_indexed(self)
        return entity


class Quaternion(numpy.ndarray):
    # unit quaternion
    def __new__(cls, w=1, x=0, y=0, z=0, n=None):
        if isinstance(x, collections.Iterable):
            n = len(x)
        elif n is None:
            n = 1
        ret = numpy.empty((n, 4) if n > 1 else 4)
        ret[..., 0] = w
        if type(x) is Vector3:
            ret[..., 1:4] = x
        else:
            ret[..., 1] = x
            ret[..., 2] = y
            ret[..., 3] = z
        return ret.view(cls)

    def __len__(self):
        if self.ndim == 1:
            return 1
        return super().__getitem__(v)

    @classmethod
    def from_angle_axis(cls, angle, axis: Vector3):
        angle = numpy.array(angle)
        angle = angle.reshape((angle.size, 1))
        n = angle.size if angle.size > 1 else len(axis)
        ret = numpy.empty((n, 4) if n > 1 else 4)
        half_angle = angle / 2
        ret[..., 0] = numpy.cos(half_angle).flatten()
        ret[..., 1:4] = axis.U * numpy.sin(half_angle)
        return ret.view(cls)

    @classmethod
    def from_direction_change(cls, before, after):
        axis = before.cross(after)
        angle = before.angle_to_vector(after)
        return cls.from_angle_axis(angle, axis)

    @property
    def I(self):
        # inverse
        ret = self.copy()
        ret[..., 1:4] *= -1
        return ret

    @property
    def w(self):
        return self[..., 0].view(numpy.ndarray)

    @property
    def xyz(self):
        return self[..., 1:4].view(Vector3)

    def split(self, n=1):
        return self.reshape(self.size // 4 // n, n, 4)

    def wxyz(self, keepdims=False):
        q = self.view(numpy.ndarray)
        if keepdims:
            return q[..., 0, numpy.newaxis], q[..., 1, numpy.newaxis], q[..., 2, numpy.newaxis], q[..., 3, numpy.newaxis]
        else:
            return q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    def mq(self, q, byrow=True):
        # byrow: multiply by row
        w, x, y, z = self.wxyz(not byrow)
        w_, x_, y_, z_ = q.wxyz(self.ndim > 2)
        shape = self.shape if byrow else (self.shape[0], q.shape[0], 4)
        ret = numpy.empty(shape)
        ret[..., 0] = w * w_ - x * x_ - y * y_ - z * z_
        ret[..., 1] = w * x_ + x * w_ + y * z_ - z * y_
        ret[..., 2] = w * y_ + y * w_ - x * z_ + z * x_
        ret[..., 3] = w * z_ + z * w_ + x * y_ - y * x_
        return ret.view(self.__class__)

    def to_angle_axis(self):
        return (
            numpy.arccos(self[..., 0]) * 2,
            self[..., 1:4].view(Vector3).U,
        )

    def to_matrix33(self):
        w, x, y, z = self.view(numpy.ndarray).T
        return numpy.array(
            [
                [
                    1 - 2 * y ** 2 - 2 * z ** 2,
                    -2 * w * z + 2 * x * y + x * y,
                    2 * w * y + 2 * x * z,
                ],
                [
                    2 * w * z + 2 * x * y,
                    1 - 2 * x ** 2 - 2 * z ** 2,
                    -2 * w * x + 2 * y * z,
                ],
                [
                    -2 * w * y + 2 * x * z,
                    2 * w * x + 2 * y * z,
                    1 - 2 * x ** 2 - 2 * y ** 2,
                ],
            ]
        ).T.view(Rotation3)


class Rotation3(Data):
    def __new__(cls, *n):
        ret = super().__new__(cls, *n, 3, 3)
        ret[:] = numpy.eye(3)
        return ret

    @classmethod
    def Rand(cls, n=1):
        return cls.Rx(numpy.random.rand(n))@cls.Ry(numpy.random.rand(n))@cls.Rz(numpy.random.rand(n))

    # rotate around body frame's axis
    @classmethod
    def from_eular_intrinsic(cls, x=0, y=0, z=0):
        return cls.Rz(z) @ cls.Ry(y) @ cls.Rx(x)

    # rotate around parent frame's axis
    @classmethod
    def from_eular_extrinsic(cls, x=0, y=0, z=0):
        return cls.Rx(x) @ cls.Ry(y) @ cls.Rz(z)

    @classmethod
    def from_angle_axis(cls, angle, axis):
        return Quaternion.from_angle_axis(angle, axis).to_matrix33().view(cls)

    @classmethod
    def Rx(cls, a, n=1):
        if isinstance(a, collections.Iterable):
            n = len(a)
        ret = numpy.full((n, 3, 3) if n > 1 else (3, 3), numpy.eye(3))
        cos = numpy.cos(a).flatten()
        sin = numpy.sin(a).flatten()
        ret[..., 1, 1] = cos
        ret[..., 1, 2] = sin
        ret[..., 2, 1] = -sin
        ret[..., 2, 2] = cos
        return ret.view(cls)

    @classmethod
    def Ry(cls, a, n=1):
        if isinstance(a, collections.Iterable):
            n = len(a)
        ret = numpy.full((n, 3, 3) if n > 1 else (3, 3), numpy.eye(3))
        cos = numpy.cos(a).flatten()
        sin = numpy.sin(a).flatten()
        ret[..., 0, 0] = cos
        ret[..., 0, 2] = -sin
        ret[..., 2, 0] = sin
        ret[..., 2, 2] = cos
        return ret.view(cls)

    @classmethod
    def Rz(cls, a, n=1):
        if isinstance(a, collections.Iterable):
            n = len(a)
        ret = numpy.full((n, 3, 3) if n > 1 else (3, 3), numpy.eye(3))
        cos = numpy.cos(a).flatten()
        sin = numpy.sin(a).flatten()
        ret[..., 0, 0] = cos
        ret[..., 0, 1] = sin
        ret[..., 1, 0] = -sin
        ret[..., 1, 1] = cos
        return ret.view(cls)

    def to_eular_extrinsic(self):
        ret = numpy.zeros((len(self), 3))
        ret[..., 0] = numpy.arctan2(self[..., 1, 2], self[..., 2, 2])
        ret[..., 1] = numpy.arcsin(-self[..., 0, 2])
        ret[..., 2] = numpy.arctan2(self[..., 0, 1], self[..., 0, 0])
        return ret

    def to_eular_intrinsic(self):
        ret = numpy.zeros((len(self), 3))
        ret[..., 0] = numpy.arctan2(self[..., 2, 1], self[..., 2, 2])
        ret[..., 1] = numpy.arcsin(-self[..., 2, 0])
        ret[..., 2] = numpy.arctan2(-self[..., 1, 0], self[..., 0, 0])
        return ret

    @property
    def I(self):
        return numpy.linalg.inv(self)

    @property
    def n(self):
        return self.shape[:-2]

    def to_matrix44(self):
        ret = numpy.full((*self.n, 4, 4), numpy.eye(4))
        ret[..., 0:3, 0:3] = self
        return ret


class Projection(Data):

    @classmethod
    def Orthogonal(cls, t, b, l, r, f, n):
        ret = numpy.eye(4)
        ret[..., 0, 0] = 2/(r-l)
        ret[..., 1, 1] = 2/(t-b)
        ret[..., 2, 2] = 2/(n-f)
        return ret.view(cls)


class Transform(Data):
    def __new__(cls, *n):
        return numpy.full(n + (4, 4), numpy.eye(4)).squeeze().view(cls)

    def __len__(self):
        if self.ndim == 2:
            return 1
        return super().__len__(self)

    @classmethod
    def Rand(cls, n=1):
        return (Vector3.Rand(n).as_scaling()@Rotation3.Rand(n).to_matrix44()@Vector3.Rand(n).as_translation()).view(cls)

    @classmethod
    def from_vector_change(cls, p0: Vector3, p1: Vector3, p0_: Vector3, p1_: Vector3):
        d = p1 - p0
        d_ = p1_ - p0_
        s = d_.norm() / d.norm()
        ret = Transform(*s.shape)
        ret.scaling = Vector3(s, s, s).squeeze()
        ret.translation = p0_ - p0
        ret.rotation = Quaternion.from_direction_change(d, d_).to_matrix33()
        return ret
    
    # rotate around body frame's axis
    @classmethod
    def from_eular_intrinsic(cls, x=0, y=0, z=0):
        return cls.Rz(z) @ cls.Ry(y) @ cls.Rx(x)

    # rotate around parent frame's axis
    @classmethod
    def from_eular_extrinsic(cls, x=0, y=0, z=0):
        return cls.Rx(x) @ cls.Ry(y) @ cls.Rz(z)
    
    @classmethod
    def Rx(cls, a, n=()):
        a=numpy.array(a)
        ret = numpy.full(n + a.shape + (4, 4), numpy.eye(4))
        cos = numpy.cos(a) 
        sin = numpy.sin(a)
        ret[..., 1, 1] = cos
        ret[..., 1, 2] = sin
        ret[..., 2, 1] = -sin
        ret[..., 2, 2] = cos
        return ret.view(cls)

    @classmethod
    def Ry(cls, a, n=()):
        a=numpy.array(a)
        ret = numpy.full(n + a.shape + (4, 4), numpy.eye(4))
        cos = numpy.cos(a) 
        sin = numpy.sin(a)
        ret[..., 0, 0] = cos
        ret[..., 0, 2] = -sin
        ret[..., 2, 0] = sin
        ret[..., 2, 2] = cos
        return ret.view(cls)

    @classmethod
    def Rz(cls, a, n=()):
        a=numpy.array(a)
        ret = numpy.full(n + a.shape + (4, 4), numpy.eye(4))
        cos = numpy.cos(a) 
        sin = numpy.sin(a)
        ret[..., 0, 0] = cos
        ret[..., 0, 1] = sin
        ret[..., 1, 0] = -sin
        ret[..., 1, 1] = cos
        return ret.view(cls)

    @property
    def n(self):
        return self.shape[:-2]

    @property
    def translation(self):
        return self[..., 3, 0:3].view(Vector3)

    @translation.setter
    def translation(self, v):
        self[..., 3, 0:3] = v

    @property
    def scaling(self):
        return numpy.linalg.norm(self[..., 0:3, 0:3], axis=1).view(Vector3)

    @scaling.setter
    def scaling(self, v):
        self[:] = (
            v.as_scaling()
            @ numpy.linalg.inv(self.scaling.as_scaling())
            @ self
        )

    @property
    def rotation(self):
        return (
            numpy.linalg.inv(self.scaling.as_scaling()[:, 0:3, 0:3])
            @ self[:, 0:3, 0:3]
        ).view(Rotation3)

    @rotation.setter
    def rotation(self, v):
        self[:] = (
            self.scaling.as_scaling()
            @ v.to_matrix44()
            @ self.translation.as_translation()
        )


class Color(Data):
    def __new__(cls, r=0, g=0, b=0, a=1, n=()):
        n = (n,) if type(n) is int else n
        if isinstance(r, collections.Iterable):
            n = *n, len(r)
        elif isinstance(g, collections.Iterable):
            n = *n, len(g)
        elif isinstance(b, collections.Iterable):
            n = *n, len(b)
        elif isinstance(a, collections.Iterable):
            n = *n, len(a)
        ret = numpy.empty((*n, 4))
        ret[..., 0] = r
        ret[..., 1] = g
        ret[..., 2] = b
        ret[..., 3] = a
        return ret.view(cls)

    def __len__(self):
        if self.ndim == 1:
            return 1
        return super().__len__()

    @classmethod
    def Rand(cls, *shape):
        ret = numpy.random.rand(*shape, 4).view(cls)
        ret.a = 1
        return ret

    @property
    def r(self):
        return self[..., 0].view(numpy.ndarray)

    @r.setter
    def r(self, v):
        force_assgin(self[..., 0], v)

    @property
    def g(self):
        return self[..., 1].view(numpy.ndarray)

    @g.setter
    def g(self, v):
        force_assgin(self[..., 1], v)

    @property
    def b(self):
        return self[..., 2].view(numpy.ndarray)

    @b.setter
    def b(self, v):
        force_assgin(self[..., 2], v)

    @property
    def a(self):
        return self[..., 3].view(numpy.ndarray)

    @a.setter
    def a(self, v):
        force_assgin(self[..., 3], v)


class Geometry(Data):
    def __new__(cls, *n):
        ret = super().__new__(cls, *n, 7)
        return ret

    @property
    def n(self):
        return self.shape[:-2] if len(self.shape) > 2 else (1,)

    @property
    def vertice(self):
        return self[..., 0:3].view(Vector3)

    @vertice.setter
    def vertice(self, v):
        force_assgin(self[..., 0:3], v)

    @property
    def color(self):
        return self[..., 3:7].view(Color)

    @color.setter
    def color(self, v):
        force_assgin(self[..., 3:7], v)


class Mesh:
    def __init__(self, *n):
        self.geometry = Geometry(*n)
        self.geometry.color = Color.Rand(*self.geometry.n)
        self.transform = Transform(*self.geometry.n)
        self.index = slice(None)

    def __getitem__(self, *n):
        ret = Mesh(*n)
        ret.geometry = self.geometry[n]
        ret.transform = self.transform[n]
        return ret

    @classmethod
    def from_indexed(cls, v):
        ret = cls(*v.n)
        ret.geometry.vertice = v
        return ret

    def p(self, i, v=None):
        if v is None:
            return self.geometry.vertice[..., i, :]
        else:
            self.geometry.vertice[..., i, :] = v

    def render(self, page=None):
        if page is None:
            page = Space(str(id(self)))
        vertice = self.geometry.vertice.mt(self.transform)
        page.render_mesh(id(self), vertice[..., self.index, :].ravel(
        ).tolist(), self.geometry.color[..., self.index, :].ravel().tolist())


class Triangle(Mesh):
    def __new__(cls, *n):
        return super().__new__(*n, 3)


class LineSegment(Geometry):
    def __new__(cls, *n):
        ret = super().__new__(cls, *n)
        ret.color = Color.Rand(*ret.n)
        return ret

    @property
    def start(self):
        return self.vertice[..., ::2, :].squeeze().view(Vector3)

    @start.setter
    def start(self, v):
        self.vertice[..., ::2, :].squeeze()[:] = v

    @property
    def end(self):
        return self.vertice[..., 1::2, :].squeeze().view(Vector3)

    @end.setter
    def end(self, v):
        self.vertice[..., 1::2, :].squeeze()[:] = v

    def render(self, page=None):
        if page is None:
            page = Space(str(id(self)))
        page.render_line(id(self), self.vertice.ravel(
        ).tolist(), self.color.ravel().tolist())


class Point(Geometry):
    def __new__(cls, *n):
        ret = super().__new__(cls, *n)
        ret.color = Color.Rand(*ret.n)
        ret.point_size = 0.1
        return ret

    def render(self, page=None):
        if page is None:
            page = Space(str(id(self)))
        page.render_point(id(self), self.vertice.ravel(
        ).tolist(), self.color.ravel().tolist(), self.point_size)


class Tetrahedron(Mesh):
    def __new__(cls, *n):
        ret = super().__new__(cls, *n, 4)
        ret.index = [0, 1, 2, 0, 2, 3, 0, 1, 3, 1, 2, 3]
        return ret


class Cube(Mesh):
    vertice_base = Vector3([
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5],
        [-0.5, -0.5, 0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [0.5, -0.5, -0.5],
        [-0.5, -0.5, -0.5]])

    def __init__(self, *n):
        super().__init__(*n, 8)
        self.geometry.vertice = self.vertice_base
        self.index = [3,
                      2,
                      0,
                      3,
                      0,
                      1,
                      5,
                      0,
                      1,
                      5,
                      4,
                      0,
                      5,
                      3,
                      1,
                      5,
                      3,
                      7,
                      6,
                      2,
                      0,
                      6,
                      4,
                      0,
                      6,
                      3,
                      2,
                      6,
                      3,
                      7,
                      6,
                      5,
                      4,
                      6,
                      5,
                      7]


class Arrow(LineSegment):
    head_base = Vector3(
        [
            [1, 0, 0],
            [0.9, 0, 0.05],
            [0.9, -0.05 * numpy.cos(pi / 6), -0.05 * numpy.sin(pi / 6)],
            [0.9, 0.05 * numpy.cos(pi / 6), -0.05 * numpy.sin(pi / 6)],
        ]
    )

    def __new__(cls, *n):
        ret = super().__new__(cls, *n, 2).view(cls)
        ret.head = Tetrahedron(*n)
        ret.head.color = ret.color[..., 0, numpy.newaxis, :]
        return ret

    def render(self, page=None):
        page = page if page else Space()
        self.head.vertice = self.head_base.mt(Transform.from_vector_change(
            Vector3(), Vector3(x=1), self.start, self.end))
        self.head.render(page)
        super().render(page)


class Plane(Data):
    def __new__(cls, position=Vector3(), normal=Vector3(z=1), n=()):
        n += merge_shapes(position.n, normal.n)
        ret = super().__new__(cls, *n, 10)
        ret.position = position
        ret.normal = normal
        return ret

    @property
    def position(self):
        return self[..., 0:3].view(Vector3)

    @position.setter
    def position(self, v):
        force_assgin(self[..., 0:3], v)

    @property
    def normal(self):
        return self[..., 3:6].view(Vector3)

    @normal.setter
    def normal(self, v):
        force_assgin(self[..., 3:6], v)

    @property
    def color(self):
        return self[..., 6:10].view(Color)

    @color.setter
    def color(self, v):
        self[..., 6:10] = v

    def as_mesh(self):
        transform = Transform.from_vector_change(
            Vector3(), Vector3(z=1), self.position, self.position+self.normal)
        return Mesh.from_indexed(Vector3.RectangleIndexed().mt(transform))

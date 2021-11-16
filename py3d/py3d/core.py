import collections
import numpy
from .server import Space

pi = numpy.arccos(-1)

def sorted_sizes(*sizes):     
    return list(dict(sorted(collections.Counter(sizes).items(),key=lambda v:v[1])).keys())
    
def merge_shapes(x,y,z):
    ret=[]
    ndim=max(len(x),len(y),len(z))
    for i in range(ndim):
        sizes = []
        sizes.append(x[-1-i]) if len(x)>i else None
        sizes.append(y[-1-i]) if len(y)>i else None
        sizes.append(z[-1-i]) if len(z)>i else None
        ret=sorted_sizes(*sizes)+ret
    return ret

def force_assgin(v1,v2):
    try:
        numpy.copyto(v1,v2)
    except:
        if v1.size == v2.size:
            numpy.copyto(v1,v2.reshape(v1.shape))
        else:
            numpy.copyto(v1,v2[:,numpy.newaxis])

class Data(numpy.ndarray):
    # usually, d1 is the number of entities, and d3 is the size of one element.  
    def __new__(cls, *shape):
        return numpy.empty(shape).view(cls)

    @property
    def n(self):
        return self.shape[:-1]

    def inflate(self, repeats, split=True):
        if self.ndim ==1:
            return numpy.full((repeats, self.shape[0]), self).view(Data)
        elif self.ndim ==2:
            if split:
                return numpy.repeat(self[:,numpy.newaxis], repeats, axis=1)
            else:
                return numpy.repeat(self, repeats, axis=0)
        elif self.ndim ==3:
            return numpy.repeat(self, repeats, axis=1)
        else:
            raise "bad data shape"

class Vector3(Data):
    def __new__(cls, x=0, y=0, z=0, n=()):
        x_ = numpy.array(x)
        if x_.ndim > 1 and x_.shape[-1] == 3:
            return x_.view(cls)
        y_ = numpy.array(y)
        z_ = numpy.array(z)
        shape=merge_shapes(x_.shape,y_.shape,z_.shape)
        n=(n,) if type(n) is int else n
        shape = *n, *shape, 3
        ret = super().__new__(cls, *shape)
        ret.x = x_
        ret.y = y_
        ret.z = z_
        return ret.view(cls)

    def __len__(self):
        if self.ndim == 1:
            return 1
        return super().__len__()

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

    @property
    def x(self):
        return self[..., 0].view(numpy.ndarray)

    @x.setter
    def x(self, v):
        force_assgin(self[..., 0],v)

    @property
    def y(self):
        return self[..., 1].view(numpy.ndarray)

    @y.setter
    def y(self, v):
        force_assgin(self[..., 1],v)

    @property
    def z(self):
        return self[..., 2].view(numpy.ndarray)

    @z.setter
    def z(self, v):
        force_assgin(self[..., 2],v)

    def norm(self) -> numpy.ndarray:  # norm
        return numpy.linalg.norm(self, axis=self.ndim - 1, keepdims=True)

    # unit vector, direction vector
    def unit(self) -> numpy.ndarray:
        n = self.norm()
        return numpy.divide(self, n, where=n != 0)

    # [1,2,3] => [3,2,1]
    def reverse(self) -> numpy.ndarray:
        self[:] = numpy.flipud(self)
        return self

    def reversed(self) -> numpy.ndarray:
        return numpy.flipud(self).copy()

    def append(self, v) -> numpy.ndarray:
        # wouldnt change self
        return numpy.vstack((self, v)).view(Vector3)

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
        return (self.as_vector4() @ v)[..., 0:3].view(self.__class__)

    def dot(self, v) -> numpy.ndarray:
        if type(v) is Vector3:
            product = self * v
            return product.sum(axis=product.ndim - 1, keepdims=True).view(numpy.ndarray)
        else:
            return numpy.dot(self, v)

    def as_vector4(self):
        return numpy.insert(self, 3, 1, axis=self.ndim-1)

    def as_scaling_matrix(self) -> numpy.ndarray:
        ret = numpy.full(self.n + (4, 4), numpy.eye(4)).squeeze()
        ret[..., 0, 0] = self[..., 0]
        ret[..., 1, 1] = self[..., 1]
        ret[..., 2, 2] = self[..., 2]
        return ret

    def as_translation_matrix(self):
        ret = numpy.full(self.n + (4, 4), numpy.eye(4)).squeeze()
        ret[..., 3, 0] = self[..., 0]
        ret[..., 3, 1] = self[..., 1]
        ret[..., 3, 2] = self[..., 2]
        return ret

    def cross(self, v: numpy.ndarray) -> numpy.ndarray:
        return numpy.cross(self, v).view(self.__class__)

    def angle_to_vector(self, to: numpy.ndarray) -> float:
        cos = self.dot(to) / self.norm() / to.norm()
        return numpy.arccos(cos)

    def angle_to_plane(self, normal: numpy.ndarray) -> float:
        return numpy.pi / 2 - self.angle_to_vector(normal)

    def rotation_to(self, to: numpy.ndarray):
        axis = self.cross(to)
        angle = self.angle_to_vector(to)
        return axis, angle

    def is_parallel_to_vector(self, v: numpy.ndarray) -> bool:
        return self.unit() == v.unit()

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

    def projection_point_on_line(
        self, p0: numpy.ndarray, p1: numpy.ndarray
    ) -> numpy.ndarray:
        return p0 + (self - p0).vector_projection(p1 - p0)

    def projection_point_on_plane(self, plane) -> numpy.ndarray:
        return self + (plane.point.expand_dims() - self).vector_projection(
            plane.normal.expand_dims()
        )

    def area(self) -> float:
        if self.ndim > 1 and self.shape[0] == 3:
            v0 = self[1] - self[0]
            v1 = self[2] - self[0]
            return v0.cross(v1).norm() / 2
        else:
            raise "size should be (3,3)"

    def as_point(self):
        point = Point(*self.n)
        point.vertice = self
        return point

    def as_vector(self,start=0):
        arrow=Arrow(*self.n)
        arrow.start=start
        arrow.end=self
        return arrow

    def as_line(self):
        vertice=numpy.repeat(self, 2, axis=self.ndim-2)[..., 1:-1, :].view(Vector3)
        line = LineSegment(*vertice.n)
        line.vertice = vertice
        return line

    def as_linesegment(self):
        line = LineSegment(*self.n)
        line.vertice = self
        return line

    def as_mesh(self):
        return Triangle.from_indexed(self)


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
        ret[..., 1:4] = axis.unit() * numpy.sin(half_angle)
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
            self[..., 1:4].view(Vector3).unit(),
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


class Transform3(Data):
    def __new__(cls, *n):
        return (
            Vector3(1, 1, 1, n=n).as_scaling_matrix()
            @ Rotation3(*n).to_matrix44()
            @ Vector3(n=n).as_translation_matrix()
        ).view(cls)

    def __len__(self):
        if self.ndim == 2:
            return 1
        return super().__len__(self)

    @classmethod
    def Rand(cls, n=1):
        return (Vector3.Rand(n).as_scaling_matrix()@Rotation3.Rand(n).to_matrix44()@Vector3.Rand(n).as_translation_matrix()).view(cls)

    @classmethod
    def from_vector_change(cls, p0: Vector3, p1: Vector3, p0_: Vector3, p1_: Vector3):
        d = p1 - p0
        d_ = p1_ - p0_
        s = d_.norm() / d.norm()
        ret = Transform3(len(s))
        ret.scaling = Vector3(s, s, s)
        ret.translation = p0_ - p0
        ret.rotation = Quaternion.from_direction_change(d, d_).to_matrix33()
        return ret

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
            v.as_scaling_matrix()
            @ numpy.linalg.inv(self.scaling.as_scaling_matrix())
            @ self
        )

    @property
    def rotation(self):
        return (
            numpy.linalg.inv(self.scaling.as_scaling_matrix()[:, 0:3, 0:3])
            @ self[:, 0:3, 0:3]
        ).view(Rotation3)

    @rotation.setter
    def rotation(self, v):
        self[:] = (
            self.scaling.as_scaling_matrix()
            @ v.to_matrix44()
            @ self.translation.as_translation_matrix()
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
        ret = numpy.random.rand(shape[0], 4).view(cls)
        ret.a = 1
        if len(shape) > 1:
            return ret[:,numpy.newaxis].repeat(shape[-1],axis=1)
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


class Entity(Data):
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

class Triangle(Entity):
    def __new__(cls, *n):
        ret=super().__new__(cls, *n, 3)
        ret.color = Color.Rand(*n, 3)
        ret.index = slice(None)
        return ret

    @classmethod
    def from_indexed(cls,v):
        ret=super().__new__(cls, *v.n)
        ret.vertice = v
        ret.color = 1
        ret.index = slice(None)
        return ret

    def p(self, i, v=None):
        if v is None:
            return self.vertice[...,i,:]
        else:
            self.vertice[...,i,:]=v

    def render(self, page=None):
        page = Space() if page is None else page
        page.render_mesh(id(self), self.vertice[...,self.index,:].flatten(
        ).tolist(), self.color[...,self.index,:].flatten().tolist())


class LineSegment(Entity):
    def __new__(cls, *n):
        ret = super().__new__(cls, *n)
        ret.color = Color.Rand(*ret.n)
        return ret

    @property
    def start(self):
        return self.vertice[...,::2,:].squeeze().view(Vector3)

    @start.setter
    def start(self,v):
        self.vertice[...,::2,:].squeeze()[:]=v

    @property
    def end(self):
        return self.vertice[...,1::2,:].squeeze().view(Vector3)

    @end.setter
    def end(self,v):
        self.vertice[...,1::2,:].squeeze()[:]=v

    def render(self, page=None):
        page = page if page else Space()
        page.render_line(id(self), self.vertice.flatten(
        ).tolist(), self.color.flatten().tolist())


class Point(Entity):
    def __new__(cls, *n):
        ret = super().__new__(cls, *n)
        ret.color = Color.Rand(*ret.n)
        ret.point_size=0.1
        return ret

    def render(self, page=None):
        page = page if page else Space()
        page.render_point(id(self), self.vertice.flatten(
        ).tolist(), self.color.flatten().tolist(), self.point_size)

class Tetrahedron(Triangle):
    def __new__(cls, n=1):
        ret = super().__new__(cls, n, 4)
        cls.index = [0, 1, 2, 0, 2, 3, 0, 1, 3, 1, 2, 3]
        return ret

    # @property
    def y(self,n):
        return self.vertice[...,n,:]


class Cube(Triangle):
    def __new__(cls, *n):
        ret = super().__new__(cls, *n, 8)
        cls.index = [0, 2, 1, 0, 4, 1, 0, 4, 2, 3, 2, 1, 5, 4, 1, 6,
                     4, 2, 5, 3, 7, 5, 3, 1, 6, 3, 7, 6, 3, 2, 6, 5, 7, 6, 5, 4]
        ret.vertice_base = Vector3(
            [
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [0.5, -0.5, 0.5],
                [-0.5, -0.5, 0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [0.5, -0.5, -0.5],
                [-0.5, -0.5, -0.5],
            ]
        )
        ret.transform = Transform3.Rand(n)
        ret.vertice = ret.vertice_base.mt(ret.transform)
        return ret


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
        self.head.vertice = self.head_base.mt(Transform3.from_vector_change(Vector3(), Vector3(x=1), self.start, self.end))
        self.head.render(page)
        super().render(page)


class Plane:
    def __init__(self, normal, point) -> None:
        self.normal = normal
        self.point = point

    @staticmethod
    def from_points(points: Vector3):
        assert len(points) >= 3
        vs = points.diff()
        normal = vs[0].cross(vs[1])
        point = points[0]
        return Plane(normal, point)

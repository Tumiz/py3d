from toweb import Color
import py3d
from typing import Optional, Tuple
import numpy
from toweb import Space
from collections import Iterable

pi = numpy.arccos(-1)


class Vector3(numpy.ndarray):
    def __new__(cls, x=0, y=0, z=0, n=1):
        if isinstance(x, Iterable):
            array = numpy.array(x)
            if array.ndim > 1 and array.shape[-1] == 3:
                return array.view(cls)
            n = len(x)
        elif isinstance(y, Iterable):
            n = len(y)
        elif isinstance(z, Iterable):
            n = len(z)
        ret = numpy.empty((n, 3))
        ret[..., 0] = x.flatten() if isinstance(x, numpy.ndarray) else x
        ret[..., 1] = y.flatten() if isinstance(x, numpy.ndarray) else y
        ret[..., 2] = z.flatten() if isinstance(x, numpy.ndarray) else z
        return ret.view(cls)

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
    def Rand(cls, n=1):
        return numpy.random.rand(n, 3).view(cls).copy()

    # construct a vector3 list of length n and filled with zero
    @classmethod
    def Zeros(cls, n: int):
        return numpy.zeros((n, 3)).view(cls).copy()

    @classmethod
    def Ones(cls, n: int):
        return numpy.ones((n, 3)).view(cls).copy()

    @property
    def x(self):
        return self[..., 0].view(numpy.ndarray)

    @x.setter
    def x(self, v):
        self[..., 0] = v

    @property
    def y(self):
        return self[..., 1].view(numpy.ndarray)

    @y.setter
    def y(self, v):
        self[..., 1] = v

    @property
    def z(self):
        return self[..., 2].view(numpy.ndarray)

    @z.setter
    def z(self, v):
        self[..., 2] = v

    def norm(self) -> numpy.ndarray:  # norm
        return numpy.linalg.norm(self, axis=self.ndim - 1, keepdims=True)

    # unit vector, direction vector
    def unit(self) -> Optional[numpy.ndarray]:
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
        return numpy.concatenate((self, v), axis=0).view(Vector3)

    def insert(self, pos, v) -> numpy.ndarray:
        # wouldnt change self
        return numpy.insert(self, pos, v, axis=0)

    def remove(self, pos) -> numpy.ndarray:
        # wouldnt change self
        return numpy.delete(self, pos, axis=0)

    def diff(self, n=1) -> numpy.ndarray:
        return numpy.diff(self, n, axis=0)

    def cumsum(self) -> numpy.ndarray:
        return super().cumsum(axis=0)

    def mq(self, q) -> numpy.ndarray:
        # multiply quaternion
        p = Quaternion(0, self)
        return (q.mq(p).mq(q.I)).xyz

    def mt(self, v) -> numpy.ndarray:
        # multiply transform
        return (self.as_vector4() @ v)[..., 0:3].view(self.__class__)

    def dot(self, v) -> numpy.ndarray:
        if type(v) is Vector3:
            return (self * v).sum(axis=1, keepdims=True).view(numpy.ndarray)
        else:
            return numpy.dot(self, v)

    def cross_matrix(self) -> numpy.ndarray:
        return len(self)

    def as_vector4(self):
        ret = numpy.ones((len(self), 4))
        ret[..., 0:3] = self
        return ret

    def as_scaling_matrix(self) -> numpy.ndarray:
        n = len(self)
        ret = numpy.full((n, 4, 4), numpy.eye(4))
        ret[..., 0, 0] = self[..., 0]
        ret[..., 1, 1] = self[..., 1]
        ret[..., 2, 2] = self[..., 2]
        return ret

    def as_translation_matrix(self):
        n = len(self)
        ret = numpy.full((n, 4, 4), numpy.eye(4))
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

    def rotation_to(self, to: numpy.ndarray) -> Tuple[numpy.ndarray, float]:
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
        return self + (plane.point - self).vector_projection(plane.normal)

    def area(self) -> float:
        if self.ndim > 1 and self.shape[0] == 3:
            v0 = self[1] - self[0]
            v1 = self[2] - self[0]
            return v0.cross(v1).norm() / 2
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

    def render_as_points(self, page=None, color=None):
        p = page if page else Space()
        p.render_points(self.flatten().tolist(), color if color else Color.Rand())
        return p

    def render_as_vector(self, origin, page=None, color=None):
        arrows = Arrow(origin, self)
        arrows.render(page, color)

    def render_as_mesh(self, page=None, color=None):
        p = page if page else Space()
        p.render_mesh(self.flatten().tolist(), color if color else Color.Rand())
        return p


class Quaternion(numpy.ndarray):
    # unit quaternion
    def __new__(cls, w=1, x=0, y=0, z=0, n=None):
        if isinstance(x, Iterable):
            n = len(x)
        elif n is None:
            n = 1
        ret = numpy.empty((n, 4))
        ret[..., 0] = w
        if type(x) is Vector3:
            ret[..., 1:4] = x
        else:
            ret[..., 1] = x
            ret[..., 2] = y
            ret[..., 3] = z
        return ret.view(cls)

    @classmethod
    def from_angle_axis(cls, angle, axis: Vector3, n=None):
        if isinstance(angle, Iterable):
            n = len(angle)
        elif len(axis) > 1:
            n = len(axis)
        elif n is None:
            n = 1
        ret = numpy.empty((n, 4))
        half_angle = angle / 2
        ret[..., 0] = numpy.cos(half_angle).flatten()
        ret[..., 1:4] = axis.unit() * numpy.sin(half_angle)
        return ret.view(cls)

    def to_angle_axis(self):
        return (
            numpy.arccos(self[..., 0]) * 2,
            self[..., 1:4].view(Vector3).unit(),
        )

    def to_matrix(self):
        w, x, y, z = self.T
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
        return self[..., 0].view(Vector1)

    @property
    def xyz(self):
        return self[..., 1:4].view(Vector3)

    def mq(self, q):
        w, x, y, z = self.T
        w_, x_, y_, z_ = q.T
        if self.shape[0] != q.shape[0]:
            ret = numpy.empty((self.shape[0], q.shape[0], 4))
            ret[..., 0] = (
                w.reshape(-1, 1) * w_
                - x.reshape(-1, 1) * x_
                - y.reshape(-1, 1) * y_
                - z.reshape(-1, 1) * z_
            )
            ret[..., 1] = (
                w.reshape(-1, 1) * x_
                + x.reshape(-1, 1) * w_
                + y.reshape(-1, 1) * z_
                - z.reshape(-1, 1) * y_
            )
            ret[..., 2] = (
                w.reshape(-1, 1) * y_
                + y.reshape(-1, 1) * w_
                - x.reshape(-1, 1) * z_
                + z.reshape(-1, 1) * x_
            )
            ret[..., 3] = (
                w.reshape(-1, 1) * z_
                + z.reshape(-1, 1) * w_
                + x.reshape(-1, 1) * y_
                - y.reshape(-1, 1) * x_
            )
        else:
            ret = numpy.empty(self.shape)
            ret[..., 0] = (w * w_ - x * x_ - y * y_ - z * z_).T
            ret[..., 1] = (w * x_ + x * w_ + y * z_ - z * y_).T
            ret[..., 2] = (w * y_ + y * w_ - x * z_ + z * x_).T
            ret[..., 3] = (w * z_ + z * w_ + x * y_ - y * x_).T
        return ret.view(self.__class__)


class Rotation3(numpy.ndarray):
    def __new__(cls, matrix=numpy.eye(3), n=1):
        return numpy.full((n, 3, 3) if n > 1 else (3, 3), matrix).view(cls)

    # rotate around body frame's axis
    @classmethod
    def from_eular_intrinsic(cls, x=0, y=0, z=0):
        return cls.Rz(z) @ cls.Ry(y) @ cls.Rx(x)

    # rotate around parent frame's axis
    @classmethod
    def from_eular_extrinsic(cls, x=0, y=0, z=0):
        return cls.Rx(x) @ cls.Ry(y) @ cls.Rz(z)

    @classmethod
    def Rx(cls, a, n=1):
        if isinstance(a, Iterable):
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
        if isinstance(a, Iterable):
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
        if isinstance(a, Iterable):
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

    def as_matrix4x4(self):
        ret = numpy.full((len(self), 4, 4) if self.ndim > 2 else (4, 4), numpy.eye(4))
        ret[..., 0:3, 0:3] = self
        return ret


class Transform3(numpy.ndarray):
    def __new__(
        cls, scale=Vector3(1, 1, 1), translation=Vector3(), rotation=Rotation3()
    ) -> None:
        return (
            scale.as_scaling_matrix()
            @ rotation.as_matrix4x4()
            @ translation.as_translation_matrix()
        ).view(cls)

    @classmethod
    def from_vector_change(cls, p0: Vector3, p1: Vector3, p0_: Vector3, p1_: Vector3):
        d = p1 - p0
        d_ = p1_ - p0_
        s = d_.norm() / d.norm()
        scale = Vector3(s, s, s)
        translation = p0_ - p0
        rotation = Quaternion.from_direction_change(d, d_).to_matrix()
        return cls(scale, translation, rotation)


class Triangle:
    def __init__(self, vertices):
        assert len(vertices) == 3
        self.vertices = vertices

    def render(self, page=""):
        p = Space(page)
        p.render_mesh(self.vertices.flatten().tolist(), Color.Rand())


class Tetrahedron:
    trianglar_index = [0, 1, 2, 0, 2, 3, 0, 1, 3, 1, 2, 3]

    def __init__(self, vertices):
        assert len(vertices) == 4
        self.vertices = vertices

    def render(self, page=""):
        p = Space(page)
        p.render_mesh(
            self.vertices[self.trianglar_index].flatten().tolist(), Color.Rand()
        )


class Cube:
    trianglar_index = [
        0,
        2,
        1,
        0,
        4,
        1,
        0,
        4,
        2,
        3,
        2,
        1,
        5,
        4,
        1,
        6,
        4,
        2,
        5,
        3,
        7,
        5,
        3,
        1,
        6,
        3,
        7,
        6,
        3,
        2,
        6,
        5,
        7,
        6,
        5,
        4,
    ]

    def __init__(self) -> None:
        self.vertices = Vector3(
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
        self.transform = Transform3()

    def render(self, s=None, color=None):
        s = Space() if not s else s
        color = Color.Rand() if not color else color
        s.render_mesh(
            self.vertices.mt(self.transform)[:, self.trianglar_index]
            .flatten()
            .tolist(),
            color,
        )


class DoubleTetrahedron:
    trianglar_index = [2, 1, 0, 2, 1, 4, 3, 1, 0, 3, 1, 4, 3, 2, 0, 3, 2, 4]

    def __init__(self, start, end):
        self.vertices = self.origin_vertices()
        self.transform = Transform3.from_vector_change(
            Vector3(), Vector3(x=1), start, end
        )

    @classmethod
    def origin_vertices(cls):
        direction = Vector3(x=1)
        length = direction.norm()
        neck_point = direction * 0.9
        neck_size = length * 0.02
        q = Quaternion.from_angle_axis(pi * 2 / 3, direction)
        p0 = neck_point + direction.cross(Vector3(z=1)).unit() * neck_size
        p1 = p0.mq(q)
        p2 = p1.mq(q)
        points = numpy.concatenate((Vector3(), p0, p1, p2, direction), axis=0)
        return points.view(Vector3)

    def render(self, space=None, color=None):
        if not space:
            space = Space()
        if not color:
            color = Color.Rand()
        space.render_mesh(
            self.vertices.mt(self.transform)[:, self.trianglar_index]
            .flatten()
            .tolist(),
            color,
        )


class Arrow:
    def __init__(self, start, end):
        self.head_vertices = Vector3(
            [
                [1, 0, 0],
                [0.9, 0, 0.05],
                [0.9, -0.05 * numpy.cos(pi / 6), -0.05 * numpy.sin(pi / 6)],
                [0.9, 0.05 * numpy.cos(pi / 6), -0.05 * numpy.sin(pi / 6)],
            ]
        ).mt(Transform3.from_vector_change(Vector3(), Vector3(x=1), start, end))
        self.line_vertices = start.insert(slice(len(end)), end)

    def render(self, space=None, color=None):
        space = space if space else Space()
        color = color if color else Color.Rand()
        space.render_mesh(
            self.head_vertices[:, Tetrahedron.trianglar_index].flatten().tolist(), color
        )
        space.render_lines(self.line_vertices.flatten().tolist(), color)


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

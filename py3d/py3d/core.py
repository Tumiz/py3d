# Copyright (c) Tumiz.
# Distributed under the terms of the GPL-3.0 License.
from __future__ import annotations
import collections
import numpy
from IPython.display import display, HTML
import pathlib
import json
import uuid

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


def render(**args):
    pwd = pathlib.Path(__file__).parent
    tmp = open(pwd/"viewer.html").read()
    html = tmp.replace("PY#D_ID", str(uuid.uuid1())).replace("PY#D_ARGS", json.dumps(args))
    if "debug" in args and args["debug"]:
        open("debug.html", "w").write(html)
    return display(HTML(html))


class Data(numpy.ndarray):
    BASE_SHAPE = ()
    # usually, d1 is the number of entities, and d3 is the size of one element.

    def __new__(cls, data: list | numpy.ndarray = [], n=()):
        return numpy.tile(data, n + (1,)).view(cls)

    @property
    def n(self):
        base_dims = len(self.BASE_SHAPE)
        if base_dims:
            return self.shape[:-base_dims]
        else:
            return self.shape

    @classmethod
    def Rand(cls, *n) -> Vector3:
        n += cls.BASE_SHAPE
        return numpy.random.rand(*n).view(cls)

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


class Vector(Data):

    def __new__(cls, data: list | numpy.ndarray = [], n=()):
        return super().__new__(cls, data, n)

    @property
    def U(self) -> Vector:
        # unit vector, direction vector
        n = self.L
        return numpy.divide(self, n, where=n != 0)

    @property
    def H(self) -> Vector:
        # Homogeneous vector
        return numpy.insert(self, self.shape[-1], 1, axis=self.ndim-1).view(Vector)

    @property
    def M(self) -> Vector:
        # mean vector
        return super().mean(axis=self.ndim-2)

    @property
    def L(self) -> Vector:
        # length
        return numpy.linalg.norm(self, axis=self.ndim - 1, keepdims=True)


class Vector2(Vector):
    BASE_SHAPE = 2,

    def __new__(cls, data: list | numpy.ndarray = [], n=()):
        return super().__new__(cls, data, n)


class Vector3(Vector):
    BASE_SHAPE = 3,

    def __new__(cls, data: list | numpy.ndarray = [], x=0, y=0, z=0, n=()):
        if data:
            return super().__new__(cls, data, n)
        else:
            x_: numpy.ndarray = numpy.array(x)
            y_: numpy.ndarray = numpy.array(y)
            z_: numpy.ndarray = numpy.array(z)
            n += merge_shapes(x_.shape, y_.shape, z_.shape)
            ret = super().__new__(cls, [0., 0., 0.], n)
            ret[..., 0] = x
            ret[..., 1] = y
            ret[..., 2] = z
            return ret

    def __matmul__(self, value: Transform) -> Vector3:
        return numpy.matmul(self.H, value)[..., 0:3].view(self.__class__)

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

    def dot(self, v) -> Vector3:
        if type(v) is Vector3:
            product = self * v
            return product.sum(axis=product.ndim - 1, keepdims=True).view(numpy.ndarray)
        else:
            return numpy.dot(self, v)

    def cross(self, v: numpy.ndarray) -> Vector3:
        return numpy.cross(self, v).view(self.__class__)

    def angle_to_vector(self, to: numpy.ndarray) -> Vector3:
        cos = self.dot(to) / self.L / to.L
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
        return self.dot(v) / v.L

    def vector_projection(self, v: numpy.ndarray) -> Vector3:
        return self.scalar_projection(v) * v / v.L

    def distance_to_line(self, p0: numpy.ndarray, p1: numpy.ndarray) -> float:
        v0 = p1 - p0
        v1 = self - p0
        return v0.cross(v1).L / v0.L

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
        i = numpy.searchsorted(xp, x).clip(1, len(xp)-1)
        x0 = xp[i-1]
        x1 = xp[i]
        d = ((x-x0)/(x1-x0))[:, numpy.newaxis]
        f0 = self[i-1]
        f1 = self[i]
        return (1-d)*f0+d*f1

    def as_scaling(self):
        ret = Transform(n=self.n)
        ret[..., 0, 0] = self[..., 0]
        ret[..., 1, 1] = self[..., 1]
        ret[..., 2, 2] = self[..., 2]
        return ret

    def as_translation(self):
        ret = Transform(n=self.n)
        ret[..., 3, 0] = self[..., 0]
        ret[..., 3, 1] = self[..., 1]
        ret[..., 3, 2] = self[..., 2]
        return ret

    def as_point(self, color=None) -> Point:
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

    def as_line(self) -> Line:
        entity = Line(*self.n)
        entity.vertice = self
        return entity

    def as_linesegment(self):
        entity = LineSegment(*self.n)
        entity.vertice = self
        return entity

    def as_mesh(self):
        entity = Mesh.from_indexed(self)
        return entity


class Transform(Data):
    BASE_SHAPE = 4, 4

    def __new__(cls, data: list | numpy.ndarray = numpy.eye(4), n=()):
        return super().__new__(cls, data, n + (1,))

    @classmethod
    def from_translation(cls, xyz_list: list | numpy.ndarray = [], x=0, y=0, z=0, n=()) -> Transform:
        return Vector3(xyz_list, x, y, z, n).as_translation()

    @classmethod
    def from_scaling(cls, xyz_list: list | numpy.ndarray = [], x=1, y=1, z=1, n=()) -> Transform:
        return Vector3(xyz_list, x, y, z, n).as_scaling()

    @classmethod
    def Rx(cls, a, n=()) -> Transform:
        a = numpy.array(a)
        ret = numpy.full(n + a.shape + (4, 4), numpy.eye(4))
        cos = numpy.cos(a)
        sin = numpy.sin(a)
        ret[..., 1, 1] = cos
        ret[..., 1, 2] = sin
        ret[..., 2, 1] = -sin
        ret[..., 2, 2] = cos
        return ret.view(cls)

    def rx(self, a, n=()) -> Transform:
        return self @ self.Rx(a, n)

    @classmethod
    def Ry(cls, a, n=()) -> Transform:
        a = numpy.array(a)
        ret = numpy.full(n + a.shape + (4, 4), numpy.eye(4))
        cos = numpy.cos(a)
        sin = numpy.sin(a)
        ret[..., 0, 0] = cos
        ret[..., 0, 2] = -sin
        ret[..., 2, 0] = sin
        ret[..., 2, 2] = cos
        return ret.view(cls)

    def ry(self, a, n=()) -> Transform:
        return self @ self.Ry(a, n)

    @classmethod
    def Rz(cls, a, n=()) -> Transform:
        a = numpy.array(a)
        ret = numpy.full(n + a.shape + (4, 4), numpy.eye(4))
        cos = numpy.cos(a)
        sin = numpy.sin(a)
        ret[..., 0, 0] = cos
        ret[..., 0, 1] = sin
        ret[..., 1, 0] = -sin
        ret[..., 1, 1] = cos
        return ret.view(cls)

    def rz(self, a, n=()) -> Transform:
        return self @ self.Rz(a, n)

    @classmethod
    def from_vector_change(cls, p0: Vector3, p1: Vector3, p0_: Vector3, p1_: Vector3) -> Transform:
        d: Vector3 = p1 - p0
        d_: Vector3 = p1_ - p0_
        s = d_.L / d.L
        ret = Transform(*s.shape)
        angle = d.angle_to_vector(d_).squeeze()
        axis = d.cross(d_)
        ret.scaling = Vector3(s, s, s).squeeze()
        ret.translation = p0_ - p0
        ret.rotation = cls.from_angle_axis(angle, axis)
        return ret

    @classmethod
    def from_angle_arbitrary_axis(cls, angle, axis_direction: Vector3, axis_point: Vector3) -> Transform:
        return Transform.from_translation(-axis_point)@Transform.from_angle_axis(angle, axis_direction)@Transform.from_translation(axis_point)

    @classmethod
    def from_angle_axis(cls, angle, axis: list | Vector3) -> Transform:
        angle = numpy.array(angle)
        axis = Vector3(axis)
        n = merge_shapes(angle.shape, axis.n)
        q = numpy.empty(n+(4,))
        half_angle = angle / 2
        q[..., 0] = numpy.cos(half_angle)
        q[..., 1:] = numpy.sin(half_angle)[..., numpy.newaxis] * axis.U
        return cls.from_quaternion(q)

    def to_angle_axis(self):
        q = self.to_quaternion()
        ha = numpy.arccos(q[..., 0])
        sin_ha = numpy.sin(ha)[..., numpy.newaxis]
        axis = numpy.divide(q[..., 1:], sin_ha,
                            where=sin_ha != 0).view(Vector3)
        return ha*2, axis

    @classmethod
    def from_quaternion(cls, quaternion: list | numpy.ndarray) -> Transform:
        q = numpy.array(quaternion)
        w = q[..., 0]
        x = q[..., 1]
        y = q[..., 2]
        z = q[..., 3]
        ret = cls(*(q.shape[:-1]))
        ret[..., 0, 0] = 1 - 2 * y ** 2 - 2 * z ** 2
        ret[..., 0, 1] = 2 * w * z + 2 * x * y
        ret[..., 0, 2] = -2 * w * y + 2 * x * z
        ret[..., 1, 0] = -2 * w * z + 2 * x * y
        ret[..., 1, 1] = 1 - 2 * x ** 2 - 2 * z ** 2
        ret[..., 1, 2] = 2 * w * x + 2 * y * z
        ret[..., 2, 0] = 2 * w * y + 2 * x * z
        ret[..., 2, 1] = -2 * w * x + 2 * y * z
        ret[..., 2, 2] = 1 - 2 * x ** 2 - 2 * y ** 2
        return ret

    def to_quaternion(self) -> ndarray:
        q = numpy.empty(self.n+(4,))
        q[..., 0] = w = (1+self[..., 0, 0]+self[..., 1, 1] +
                         self[..., 2, 2])**0.5/2
        q[..., 1] = numpy.divide(
            self[..., 1, 2]-self[..., 2, 1], 4*w, where=w != 0)
        q[..., 2] = numpy.divide(
            self[..., 2, 0]-self[..., 0, 2], 4*w, where=w != 0)
        q[..., 3] = numpy.divide(
            self[..., 0, 1]-self[..., 1, 0], 4*w, where=w != 0)
        return q

    @classmethod
    def from_euler(cls, sequence: str, angles_list: list | numpy.ndarray) -> Transform:
        lo = sequence.lower()
        v = numpy.array(angles_list)
        m = {a: getattr(cls, "R" + a.lower()) for a in 'xyz'}
        if sequence.islower():
            return m[lo[0]](v[..., 0]) @ m[lo[1]](v[..., 1]) @ m[lo[2]](v[..., 2])
        else:
            return m[lo[2]](v[..., 2]) @ m[lo[1]](v[..., 1]) @ m[lo[0]](v[..., 0])

    def to_euler(self, sequence: str):
        extrinsic = sequence.islower()
        lo = sequence.lower()
        ret = numpy.zeros((*self.n, 3))
        i = [0, 1, 2] if extrinsic else [2, 1, 0]
        m = [0 if o == 'x' else 1 if o == 'y' else 2 for o in lo]
        if not extrinsic:
            m.reverse()

        def f(x, y): return -1 if x-y == 2 or x - y == -1 else 1
        a = [f(m[1], m[0]), f(m[2], m[0]), f(m[2], m[1])]
        if a.count(-1) > 1:
            b = [-1, 1, 1, -1, 1]
        else:
            b = [1, a[0], a[1], 1, a[2]]

        ret[..., i[0]] = numpy.arctan2(
            b[0]*self[..., m[1], m[2]], b[1]*self[..., 3-m[0]-m[1], m[2]])
        ret[..., i[1]] = getattr(numpy, 'arccos' if m[0] == m[2] else 'arcsin')(
            b[2]*self[..., m[0], m[2]])
        ret[..., i[2]] = numpy.arctan2(
            b[3]*self[..., m[0], m[1]], b[4]*self[..., m[0], 3-m[1]-m[2]])
        return ret

    @classmethod
    def from_rpy(cls, angles_list: list | numpy.ndarray) -> Transform:
        return cls.from_euler('XYZ', angles_list)

    def to_rpy(self):
        return self.to_euler('XYZ')

    @property
    def translation_vector(self) -> Vector3:
        return self[..., 3, 0:3].view(Vector3)

    @translation_vector.setter
    def translation_vector(self, v: Vector3):
        self[..., 3, 0:3] = v

    @property
    def scaling_vector(self) -> Vector3:
        return numpy.linalg.norm(self[..., 0:3, 0:3], axis=self.ndim-1).view(Vector3)

    @scaling_vector.setter
    def scaling_vector(self, v: Vector3):
        self[:] = v.as_scaling() @ self.scaling.I @ self

    @property
    def translation(self) -> Transform:
        ret = Transform(*self.n)
        ret[..., 3, 0:3] = self.translation_vector
        return ret

    @translation.setter
    def translation(self, v: numpy.ndarray):
        self[..., 3, :3] = v[..., 3, :3]

    @property
    def scaling(self) -> Transform:
        ret = Transform(*self.n)
        ret[..., range(3), range(3)] = self.scaling_vector
        return ret

    @scaling.setter
    def scaling(self, v: numpy.ndarray):
        self[:] = v @ self.scaling.I @ self

    @property
    def rotation(self) -> Transform:
        return self.scaling.I @ self @ self.translation.I

    @rotation.setter
    def rotation(self, v):
        self[:] = self.scaling @ v @ self.translation

    @property
    def I(self) -> Transform:
        return numpy.linalg.inv(self)

    @property
    def forward(self) -> Vector3:
        return Vector3(x=1).mt(self)

    @classmethod
    def from_perspective(cls, fovy, aspect, near, far) -> Transform:
        f = 1 / numpy.tan(fovy/2)
        range_inv = 1.0 / (near - far)
        return numpy.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (near + far) * range_inv, 1],
            [0, 0, -2 * near * far * range_inv, 0]
        ])

    @classmethod
    def from_orthographic(cls, l, r, t, b, n, f):
        pass

    def interp(self, xp, x) -> Transform:
        xp = numpy.array(xp)
        x = numpy.array(x)
        i = numpy.searchsorted(xp, x).clip(1, len(xp)-1)
        x0 = xp[i-1]
        x1 = xp[i]
        d: numpy.ndarray = (x-x0)/(x1-x0)
        r0: Transform = self.rotation[i-1]
        r1: Transform = self.rotation[i]
        t0: Vector3 = self.translation_vector[i-1]
        t1: Vector3 = self.translation_vector[i]
        s0: Vector3 = self.scaling_vector[i-1]
        s1: Vector3 = self.scaling_vector[i]
        angle, axis = (r0.I@r1).to_angle_axis()
        rotation = Transform.from_angle_axis(d*angle, axis)
        translation = (d[..., numpy.newaxis] * (t1 - t0)).as_translation()
        scaling = (d[..., numpy.newaxis] * s1 / s0).as_scaling()
        return self[i-1]@scaling@rotation@translation


class Color(Vector):
    BASE_SHAPE = (4,)

    def __new__(cls, data: numpy.ndarray | list = [], r=0, g=0, b=0, a=1, n=()):
        if data:
            return super().__new__(cls, data, n)
        else:
            r_: numpy.ndarray = numpy.array(r)
            g_: numpy.ndarray = numpy.array(g)
            b_: numpy.ndarray = numpy.array(b)
            a_: numpy.ndarray = numpy.array(a)
            n += merge_shapes(r_.shape, g_.shape, b_.shape, a_.shape)
            ret = super().__new__(cls, [0, 0, 0, 1], n)
            ret[..., 0] = r
            ret[..., 1] = g
            ret[..., 2] = b
            ret[..., 3] = a
        return ret.view(cls)

    def __len__(self):
        if self.ndim == 1:
            return 1
        return super().__len__()

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
        self[..., 3] = v


class Point(Vector):
    BASE_SHAPE = 7,

    def __new__(cls, *n):
        ret = super().__new__(cls, [0.] * 7, n)
        ret.color = Color.Rand(*n[:-1], 1)
        ret.color.a = 1
        ret.type = "POINTS"
        return ret

    @property
    def vertice(self):
        return self[..., 0:3].view(Vector3)

    @vertice.setter
    def vertice(self, v):
        self[..., 0:3] = v

    @property
    def color(self):
        return self[..., 3:7].view(Color)

    @color.setter
    def color(self, v):
        self[..., 3:7] = v

    def render(self, **args):
        render(id=id(self), mode=self.type, vertice=self.vertice.ravel(
        ).tolist(), color=self.color.ravel().tolist(), **args)


class Triangle(Point):
    def __new__(cls, *n):
        ret = super().__new__(cls, *n, 3)
        ret.type = "TRIANGLES"
        return ret


class LineSegment(Point):
    def __new__(cls, *n):
        ret = super().__new__(cls, *n)
        ret.type = "LINES"
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


class Line(Point):
    def __new__(cls, *n):
        ret = super().__new__(cls, *n)
        ret.type = "LINE_STRIP"
        return ret

class Mesh:
    def __init__(self, *n):
        self.geometry = Point(*n)
        self.geometry.color = Color.Rand(*self.geometry.n)
        self.transform = Transform(n=self.geometry.n)
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

    def render(self, **args):
        render(id=id(self), mode="TRIANGLES", vertice=(self.geometry.vertice @ self.transform).ravel(
        ).tolist(), color=self.geometry.color.ravel().tolist(), **args)
        
class Camera(Mesh):
    def __init__(self, *n):
        super().__init__(*n)
        self.projection = Transform()

    def set_perspective(self, fov, aspect, near, far):
        self.projection = Transform.from_perspective(fov, aspect, near, far)

    @property
    def matrix(self):
        return self.transform.I @ self.projection


class Tetrahedron(Mesh):
    def __init__(self, *n):
        super().__init__(*n, 4)
        self.index = [0, 1, 2, 0, 2, 3, 0, 1, 3, 1, 2, 3]


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

# Copyright (c) Tumiz.
# Distributed under the terms of the GPL-3.0 License.
from __future__ import annotations
import numpy
from IPython.display import display, update_display, HTML
from typing import Dict
import pathlib
import uuid
import json

pi = numpy.arccos(-1)


def sign(v):
    return numpy.sign(v, where=v != 0, out=numpy.ones_like(v))


class Viewer:
    tmp = open(pathlib.Path(__file__).parent/"viewer.html").read()

    def __init__(self) -> None:
        self.cache : Dict[float, list] = {}
        self.id = str(uuid.uuid1())
        display(HTML(""), display_id=self.id)

    def render_args(self, t, **args):
        if t in self.cache:
            self.cache[t].append(args)
        else:
            self.cache[t] = [args]

    def show(self, **args):
        html = self.tmp.replace("PY#D_ID", self.id).replace(
            "PY#D_ARGS", json.dumps(self.cache))
        if "debug" in args and args["debug"]:
            open(self.id+".html", "w").write(html)
        update_display(HTML(html), display_id=self.id)

    def render(self, *objs: Point, t=0, **args):
        for obj in objs:
            self.render_args(mode=obj.TYPE, t=t, vertex=obj.vertex.ravel(
            ).tolist(), color=obj.color.ravel().tolist(), **args)


class Data(numpy.ndarray):
    BASE_SHAPE = ()

    def __new__(cls, data: list | numpy.ndarray = [], n=()):
        return numpy.tile(data, n + (1,)).view(cls)

    def __imatmul__(self, value) -> Data:
        return self @ value

    @property
    def n(self):
        base_dims = len(self.BASE_SHAPE)
        if base_dims:
            return self.shape[:-base_dims]
        else:
            return self.shape

    @classmethod
    def Rand(cls, *n) -> Data:
        n += cls.BASE_SHAPE
        return numpy.random.rand(*n).view(cls)

    @classmethod
    def load(cls, path) -> Data:
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
        if numpy.any(data):
            return super().__new__(cls, data, n)
        else:
            n += max(numpy.shape(x), numpy.shape(y), numpy.shape(z))
            ret = super().__new__(cls, [0., 0., 0.], n)
            ret.x = x
            ret.y = y
            ret.z = z
            return ret

    @classmethod
    def Grid(cls, x=0, y=0, z=0, n=()) -> Vector3:
        n += numpy.shape(x) + numpy.shape(y) + numpy.shape(z)
        ret = super().__new__(cls, [0., 0., 0.], n)
        i = numpy.arange(len(n))
        ret.x = numpy.expand_dims(x, axis=i[i != 0].tolist())
        ret.y = numpy.expand_dims(y, axis=i[i != 1].tolist())
        ret.z = numpy.expand_dims(z, axis=i[i != 2].tolist())
        return ret

    def __matmul__(self, value: Transform) -> Vector3:
        if type(value) is Transform:
            return numpy.matmul(self.H, value)[..., 0:3].view(Vector3)
        else:
            return self.__matmul__(value)

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
        entity.vertex = self
        if color is not None:
            entity.color = color
        return entity

    def as_line(self) -> Line:
        entity = Line(*self.n)
        entity.vertex = self
        return entity

    def as_linesegment(self) -> LineSegment:
        entity = LineSegment(*self.n)
        entity.vertex = self
        return entity

    def as_triangle(self):
        entity = Triangle(*self.n[:-1])
        entity.vertex = self
        return entity

    def as_mesh(self):
        entity = Mesh.from_indexed(self)
        return entity


class Vector4(Vector):
    BASE_SHAPE = 4,

    def __new__(cls, xyzw_list: list | numpy.ndarray = [0., 0., 0., 1.], n=()):
        return super().__new__(cls, xyzw_list, n)

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

    @property
    def w(self):
        return self[..., 3].view(numpy.ndarray)

    @w.setter
    def w(self, v):
        self[..., 3] = v

    @property
    def xyz(self) -> Vector3:
        return self[..., 0:3].view(Vector3)

    @xyz.setter
    def xyz(self, v):
        self[..., 0:3] = v

    @property
    def wxyz(self) -> Vector:
        ret = Vector([0, 0, 0, 0], n=self.n)
        ret[..., 0] = self.w
        ret[..., 1:4] = self.xyz
        return ret

    def as_transform(self) -> Transform:
        return Transform.from_quaternion(self)


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
    def from_vector_change(cls, a: Vector3, b: Vector3) -> Transform:
        ret = Transform(max(a.shape, b.shape))
        angle = a.angle_to_vector(b).squeeze()
        axis = a.cross(b)
        ret.rotation = cls.from_angle_axis(angle, axis)
        return ret

    @classmethod
    def from_angle_axis(cls, angle, axis: list | Vector3) -> Transform:
        angle = numpy.array(angle)
        axis = Vector3(axis)
        n = max(angle.shape, axis.n)
        q = numpy.empty(n+(4,))
        half_angle = angle / 2
        q[..., 0] = numpy.cos(half_angle)
        q[..., 1:] = numpy.sin(half_angle)[..., numpy.newaxis] * axis.U
        return cls.from_quaternion(q)

    def as_angle_axis(self):
        q = self.as_quaternion()
        ha = numpy.arccos(q[..., 0])
        sin_ha = numpy.sin(ha)[..., numpy.newaxis]
        axis = numpy.divide(q[..., 1:], sin_ha,
                            where=sin_ha != 0).view(Vector3)
        return ha*2, axis

    @classmethod
    def from_quaternion(cls, xyzw_list: list | numpy.ndarray) -> Transform:
        q = Vector4(xyzw_list)
        ret = cls(n=q.shape[:-1])
        ret[..., 0, 0] = 1 - 2 * q.y ** 2 - 2 * q.z ** 2
        ret[..., 0, 1] = 2 * q.w * q.z + 2 * q.x * q.y
        ret[..., 0, 2] = -2 * q.w * q.y + 2 * q.x * q.z
        ret[..., 1, 0] = -2 * q.w * q.z + 2 * q.x * q.y
        ret[..., 1, 1] = 1 - 2 * q.x ** 2 - 2 * q.z ** 2
        ret[..., 1, 2] = 2 * q.w * q.x + 2 * q.y * q.z
        ret[..., 2, 0] = 2 * q.w * q.y + 2 * q.x * q.z
        ret[..., 2, 1] = -2 * q.w * q.x + 2 * q.y * q.z
        ret[..., 2, 2] = 1 - 2 * q.x ** 2 - 2 * q.y ** 2
        return ret

    def as_quaternion(self) -> Vector4:
        q = Vector4(n=self.n)
        q.w = numpy.sqrt(1+self[..., 0, 0]+self[..., 1, 1] + self[..., 2, 2])/2
        m0 = self[q.w == 0]
        m1, w1 = self[q.w != 0], q.w[q.w != 0]
        q.x[q.w != 0] = numpy.divide(m1[..., 1, 2]-m1[..., 2, 1], 4*w1)
        q.y[q.w != 0] = numpy.divide(m1[..., 2, 0]-m1[..., 0, 2], 4*w1)
        q.z[q.w != 0] = numpy.divide(m1[..., 0, 1]-m1[..., 1, 0], 4*w1)
        q.x[q.w == 0] = sign(m0[..., 1, 2]-m0[..., 2, 1]) * numpy.sqrt(
            1+m0[..., 0, 0]-m0[..., 1, 1] - m0[..., 2, 2])/2
        q.y[q.w == 0] = sign(m0[..., 2, 0]-m0[..., 0, 2]) * numpy.sqrt(
            1-m0[..., 0, 0]+m0[..., 1, 1] - m0[..., 2, 2])/2
        q.z[q.w == 0] = sign(m0[..., 0, 1]-m0[..., 1, 0]) * numpy.sqrt(
            1-m0[..., 0, 0]-m0[..., 1, 1] + m0[..., 2, 2])/2
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

    def as_euler(self, sequence: str):
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

    def as_rpy(self):
        return self.as_euler('XYZ')

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
        angle, axis = (r0.I@r1).as_angle_axis()
        rotation = Transform.from_angle_axis(d*angle, axis)
        translation = (d[..., numpy.newaxis] * (t1 - t0)).as_translation()
        scaling = (d[..., numpy.newaxis] * s1 / s0).as_scaling()
        return self[i-1]@scaling@rotation@translation


class Color(Vector):
    BASE_SHAPE = 4,

    def __new__(cls, data: numpy.ndarray | list = [], r=0, g=0, b=0, a=1, n=()):
        if numpy.any(data):
            return super().__new__(cls, data, n)
        else:
            n += max(numpy.shape(r), numpy.shape(g),
                     numpy.shape(b), numpy.shape(a))
            ret = super().__new__(cls, [0, 0, 0, 1], n)
            ret[..., 0] = r
            ret[..., 1] = g
            ret[..., 2] = b
            ret[..., 3] = a
        return ret

    @property
    def r(self):
        return self[..., 0].view(numpy.ndarray)

    @r.setter
    def r(self, v):
        self[..., 0] = v

    @property
    def g(self):
        return self[..., 1].view(numpy.ndarray)

    @g.setter
    def g(self, v):
        self[..., 1] = v

    @property
    def b(self):
        return self[..., 2].view(numpy.ndarray)

    @b.setter
    def b(self, v):
        self[..., 2] = v

    @property
    def a(self):
        return self[..., 3].view(numpy.ndarray)

    @a.setter
    def a(self, v):
        self[..., 3] = v


class Point(Data):
    BASE_SHAPE = 7,
    TYPE = "POINTS"

    def __new__(cls, *n):
        ret = numpy.empty(n + cls.BASE_SHAPE).view(cls)
        ret.color = Color.Rand(*(n[:-1] if len(cls.BASE_SHAPE) == 1 else n), 1)
        ret.color.a = 1
        return ret

    @property
    def vertex(self) -> Vector3:
        return self[..., 0:3].view(Vector3)

    @vertex.setter
    def vertex(self, v):
        self[..., 0:3] = v

    @property
    def color(self) -> Color:
        return self[..., 3:7].view(Color)

    @color.setter
    def color(self, v):
        self[..., 3:7] = v

    def render(self, **args):
        v = Viewer()
        v.render(self, **args)
        v.show(**args)


class Triangle(Point):
    BASE_SHAPE = 3, 7
    TYPE = "TRIANGLES"

    def __new__(cls, *n):
        ret = super().__new__(cls, *n)
        return ret


class LineSegment(Point):
    TYPE = "LINES"

    def __new__(cls, *n):
        ret = super().__new__(cls, *n)
        return ret

    @property
    def start(self):
        return self[..., ::2, :].view(Point)

    @property
    def end(self):
        return self[..., 1::2, :].view(Point)


class Line(Point):
    TYPE = "LINE_STRIP"

    def __new__(cls, *n):
        ret = super().__new__(cls, *n)
        return ret


class Mesh(Point):
    TYPE = "TRIANGLES"

    def __new__(cls, *n):
        ret = super().__new__(cls, *n)
        return ret

    @classmethod
    def from_indexed(cls, v):
        ret = cls(*v.n)
        ret.vertex = v
        return ret


class Camera:
    def __init__(self, *n):
        self.transform = Transform(*n)
        self.projection = Transform(*n)

    def set_perspective(self, fov, aspect, near, far):
        self.projection = Transform.from_perspective(fov, aspect, near, far)

    @property
    def matrix(self):
        return self.transform.I @ self.projection


class Utils:
    car_points: Vector3 = Vector3.load(pathlib.Path(__file__).parent/"car.npy")

    @classmethod
    def Car(cls, wire_frame=False) -> Point:
        if wire_frame:
            ret = Vector3([
                [-1, 1, 0],
                [-1, 1, 0.8],
                [0, 1, 1.6],
                [2, 1, 1.6],
                [4, 1, 0.8],
                [4, 1, 0],
                [-1, -1, 0],
                [-1, -1, 0],
                [-1, -1, 0.8],
                [0, -1, 1.6],
                [2, -1, 1.6],
                [4, -1, 0.8],
                [4, -1, 0],
                [-1, -1, 0]
            ])
            index = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0, 6, 7, 7, 8, 8, 9, 9,
                     10, 10, 11, 11, 12, 12, 6, 0, 7, 1, 8, 2, 9, 3, 10, 4, 11, 5, 12]
            return ret[index].as_linesegment()
        else:
            return cls.car_points.as_point()

    @classmethod
    def Grid(cls, size=5) -> LineSegment:
        v = Vector3(x=[-size, size]) @ Transform.from_translation(
            y=range(-size, size+1)) @ Transform.from_rpy([[[0, 0, 0]], [[0, 0, pi/2]]])
        l = v.as_linesegment()
        l.color = Color()
        l.color[0, size] = Color(r=[0, 1])
        l.color[1, size] = Color(g=[0, 1])
        return l

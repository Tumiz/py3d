# Copyright (c) Tumiz.
# Distributed under the terms of the GPL-3.0 License.
from __future__ import annotations
import PIL.Image
import numpy
from IPython.display import display, HTML, clear_output
from typing import Dict
import pathlib
import json
import struct
import multiprocessing
import http.server
import socket

pi = numpy.arccos(-1)
__module__ = __import__(__name__)
numpy.set_printoptions(linewidth=600)


def sign(v):
    return numpy.sign(v, where=v != 0, out=numpy.ones_like(v))


def launch_server(ip, port):
    server = http.server.HTTPServer(
        (ip, port), http.server.SimpleHTTPRequestHandler)
    server.serve_forever()


class View:
    __preload__ = open(pathlib.Path(__file__).parent/"viewer.html").read()

    def __init__(self) -> None:
        self.cache: Dict[float, list] = {}
        self.min = []
        self.max = []
        self.viewpoint = None
        self.lookat = None
        self.up = None
        self.size = (600, 1000)

    def __render_args__(self, t, **args):
        t = round(t, 3)
        if t in self.cache:
            self.cache[t].append(args)
        else:
            self.cache[t] = [args]
        return self

    def _repr_html_(self):
        html = self.__preload__.replace(
            "PY#D_ARGS", json.dumps(self.__dict__))
        self.cache.clear()
        self.max = []
        self.min = []
        self.viewpoint = None
        self.lookat = None
        self.up = None
        self.size = (600, 1000)
        return html

    def show(self, viewpoint=None, lookat=None, up=None, inplace=True, size=[], in_jupyter=True, name="py3d", port=9871):
        '''
        same as py3d.show
        '''
        self.viewpoint = viewpoint
        self.lookat = lookat
        self.up = up
        self.size = size
        if in_jupyter:
            if inplace:
                clear_output(True)
            return display(HTML(self._repr_html_()))
        else:
            index = f"{name}.html"
            open(index, "w").write(self._repr_html_())
            sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sk.connect(("10.255.255.255", 1))
                ip = sk.getsockname()[0]
            except:
                ip = "127.0.0.1"
            finally:
                sk.close()
            print(f"click http://{ip}:{port}/{index} to view")
            multiprocessing.Process(
                target=launch_server, args=(ip, port), daemon=True).start()

    def render(self, obj: Point, t=0):
        if obj.any():
            if self.max == []:
                self.max = obj.xyz.flatten().max(-2).tolist()
                self.min = obj.xyz.flatten().min(-2).tolist()
            else:
                self.max = numpy.max(
                    [self.max, obj.xyz.flatten().max(-2)], axis=0).tolist()
                self.min = numpy.min(
                    [self.min, obj.xyz.flatten().min(-2)], axis=0).tolist()
            return self.__render_args__(t=t, mode=obj.TYPE, vertex=obj.xyz.ravel(
            ).tolist(), color=obj.color.ravel().tolist(), normal=obj.normal.ravel().tolist())

    def label(self, text: str, position: list = [0, 0, 0], color="grey", t=0):
        return self.__render_args__(t=t, mode="TEXT", text=text,
                                    vertex=position, color=color)


default_view = View()


def render(*objs, t=0):
    for obj in objs:
        default_view.render(obj, t)
    return default_view


def label(text, position: list = [0, 0, 0], color="grey", t=0):
    return default_view.label(text, position, color, t)


def show(viewpoint=None, lookat=None, up=None, inplace=True, size=[600, 1000], in_jupyter=True, name="py3d", port=9871):
    '''
    display all rendered objects in one scene
    viewpoint: the position from where to view the scene
    lookat: the position to look at
    up: up direction to view the scene
    inplace: update the output when displayed in jupyter
    size: size of the viewer, (height, width)
    in_jupyter: display in jupyter, as a output, otherwise in a web browser
    name: name of the page when displayed in a web browser
    port: port to visit the page when displayed in a web browser
    '''
    return default_view.show(viewpoint, lookat, up, inplace, size, in_jupyter, name, port)


def read_img(path) -> Vector:
    return Vector(PIL.Image.open(path))


def read_pcd(path) -> Vector:
    f = open(path, "rb")
    data_type = ""
    ret = []
    tp_map = {
        (1, "I"): "b",
        (1, "U"): "B",
        (2, "I"): "h",
        (2, "U"): "H",
        (4, "I"): "i",
        (4, "U"): "I",
        (4, "F"): "f",
        (8, "I"): "q",
        (8, "U"): "Q",
        (8, "F"): "d"
    }
    while True:
        if "binary" not in data_type:
            r = str(f.readline(), encoding="utf-8")[:-1]
            if not r:
                break
            elif r.startswith("FIELDS"):
                cols = r.replace("FIELDS ", "").split(" ")
            elif r.startswith("SIZE"):
                size = [int(s) for s in r.replace("SIZE ", "").split(" ")]
            elif r.startswith("TYPE"):
                tp = r.replace("TYPE ", "").split(" ")
            elif r.startswith("POINTS"):
                count = int(r.replace("POINTS ", ""))
            elif r.startswith("DATA"):
                data_type = r.replace("DATA ", "")
            elif data_type == "ascii":
                ret.append([float(s) for s in r.split(" ")])
        elif "compressed" in data_type:
            raise Exception(f"{data_type} PCD is not currently supported")
        else:
            if count > 0:
                count -= 1
            else:
                break
            tmp = []
            for s, t in zip(size, tp):
                bt = f.read(s)
                d, = struct.unpack(tp_map[(s, t)], bt)
                tmp.append(d)
            ret.append(tmp)
    ret = Vector(ret)
    for i, c in enumerate(cols):
        setattr(ret, c, ret[:, i])
    return ret


def read_ply(path) -> tuple[Vector3, Triangle]:
    f = open(path, "rb")
    n_vertex = 0
    n_face = 0
    for c in f:
        c = c.strip().decode("utf-8")
        if "format" in c:
            data_type = c.split(" ", 1)[1]
        elif "element vertex" in c:
            n_vertex = int(c.split(" ")[-1])
        elif "element face" in c:
            n_face = int(c.split(" ")[-1])
        elif "end_header" in c:
            break
    vertices = []
    while n_vertex:
        if "ascii" in data_type:
            c = f.readline().decode("utf-8").strip()
            vertices.append([float(v) for v in c.split(" ")])
        else:
            vertices.append([struct.unpack("f", f.read(4))[0]
                            for i in range(3)])
        n_vertex -= 1
    faces = []
    while n_face:
        if "ascii" in data_type:
            c = f.readline().decode("utf-8").strip()
            size, *d = [int(v) for v in c.split(" ")]
        else:
            size, = struct.unpack("B", f.read(1))
            d = struct.unpack("<" + "i" * size, f.read(4 * size))
        if size > 3:
            tmp = []
            for i in range(size-2):
                tmp.append([d[0], d[i+1], d[i+2]])
            d = tmp
        faces.append(d)
        n_face -= 1
    vertices = Vector3(vertices)
    triangles: Vector3 = vertices[faces]
    a = triangles[..., 0, :]
    b = triangles[..., 1, :]
    c = triangles[..., 2, :]
    normals = (b - a).cross(c - b).U
    mesh = numpy.empty((numpy.prod(triangles.shape[:-1]), 10)).view(Triangle)
    mesh.xyz = triangles.flatten()
    mesh.color = Color.standard((1,))
    mesh.normal = normals.flatten().repeat(3, axis=-2)
    return vertices, mesh


def read_csv(path, header=0) -> Vector:
    '''
    Load data from a csv file. 
    header: line number of header, 0 if there is no header
    '''
    ret = numpy.loadtxt(path, delimiter=',', skiprows=header).view(Vector)
    with open(path) as f:
        for i in range(header):
            ret.columns = f.readline().strip().split(",")
    return ret


def read_txt(path, delimiter=' ', **args) -> Vector:
    '''
    Load data from a text file
    '''
    return numpy.loadtxt(path, delimiter=delimiter, **args).view(Vector)


def read_npy(path) -> Vector:
    return numpy.load(path).view(Vector)


def rand(*n) -> Vector | Vector2 | Vector3 | Vector4:
    '''
    Create a random vector with shape of n
    '''
    w = n[-1]
    if w in [2, 3, 4]:
        vtype = getattr(__module__, f"Vector{w}")
    else:
        vtype = Vector
    return numpy.random.rand(*n).view(vtype)


class Vector(numpy.ndarray):
    '''
    Base class of Vector2, Vector3, Vector4 and Transform
    '''
    BASE_SHAPE = ()

    def __new__(cls, data: list | numpy.ndarray = [], columns=[]):
        nd = numpy.array(data)
        if cls.BASE_SHAPE:
            bn = len(cls.BASE_SHAPE)
            ret = numpy.zeros(nd.shape[:-bn]+cls.BASE_SHAPE)
            c = numpy.minimum(cls.BASE_SHAPE, nd.shape[-bn:])
            mask = ..., *[slice(s) for s in c]
            ret[mask] = nd[mask]
            return ret.view(cls)
        else:
            ret = nd.view(cls)
            ret.columns = []
            for c in columns:
                ret.columns.append(c)
            return ret

    def __imatmul__(self, value) -> Vector:
        return self @ value

    def __getitem__(self, keys) -> Vector:
        if hasattr(self, "columns"):
            if isinstance(keys, str):
                keys = ..., self.columns.index(keys)
            elif isinstance(keys, tuple) and all(isinstance(k, str) for k in keys):
                keys = ..., [self.columns.index(key) for key in keys]
        return super().__getitem__(keys)

    def tile(self, *n) -> Vector:
        return numpy.tile(self, n + self.ndim * (1,))

    def flatten(self, base_shape=None) -> Vector:
        '''
        Return a copy of the vector reshaped into (-1, *base_shape).
        '''
        if base_shape is None:
            base_shape = self.BASE_SHAPE if self.BASE_SHAPE else self.shape[-1:]
        return self.reshape(-1, *base_shape)

    def sample(self, n, base_shape=None):
        '''
        Return a random sample of items
        '''
        flattened = self.flatten(base_shape)
        indices = numpy.random.randint(0, len(flattened), n)
        return flattened[indices]

    @property
    def n(self):
        base_dims = len(self.BASE_SHAPE)
        if base_dims:
            return self.shape[:-base_dims]
        else:
            return self.shape

    @property
    def x(self) -> Vector:
        return self[..., 0].view(Vector)

    @x.setter
    def x(self, v):
        self[..., 0] = v

    @property
    def y(self) -> Vector:
        return self[..., 1].view(Vector)

    @y.setter
    def y(self, v):
        self[..., 1] = v

    @property
    def z(self) -> Vector:
        return self[..., 2].view(Vector)

    @z.setter
    def z(self, v):
        self[..., 2] = v

    @property
    def w(self):
        return self[..., 3].view(Vector)

    @w.setter
    def w(self, v):
        self[..., 3] = v

    @property
    def xy(self) -> Vector2:
        return self[..., 0:2].view(Vector2)

    @xy.setter
    def xy(self, v):
        self[..., 0:2] = v

    @property
    def yz(self) -> Vector2:
        return self[..., 1:3].view(Vector2)

    @yz.setter
    def yz(self, v):
        self[..., 1:3] = v

    @property
    def xyz(self) -> Vector3:
        return self[..., 0:3].view(Vector3)

    @xyz.setter
    def xyz(self, v):
        self[..., 0:3] = v

    @property
    def U(self) -> Vector | Vector2 | Vector3 | Vector4:
        '''
        unit vector, direction vector
        '''
        l = numpy.linalg.norm(self, axis=self.ndim - 1, keepdims=True)
        return numpy.divide(self, l, where=l != 0)

    @property
    def H(self) -> Vector | Vector2 | Vector3 | Vector4:
        '''
        Homogeneous vector
        '''
        ret = numpy.insert(self, self.shape[-1], 1, axis=self.ndim-1)
        w = ret.shape[-1]
        if w in [2, 3, 4]:
            return ret.view(getattr(__module__, f"Vector{w}"))
        else:
            return ret.view(Vector)

    @property
    def M(self) -> Vector | Vector2 | Vector3 | Vector4:
        # mean vector
        return super().mean(axis=self.ndim-2)

    @property
    def L(self) -> numpy.ndarray:
        # length
        return numpy.linalg.norm(self, axis=self.ndim - 1)

    def diff(self, n=1) -> Vector:
        return numpy.diff(self, n, axis=self.ndim-2)

    def lerp(self, target_x, origin_x) -> Vector:
        '''
        Linear interpolation
        target_x: 1-D array, the series to be interpolated. For example, time series.
        origin_x: 1-D array, the series to interpolate 'target_x' into, with same length as self. 
        Only translation, rotation and scaling can be interpolated
        '''
        x = numpy.array(target_x)
        xp = numpy.array(origin_x)
        assert x.ndim <= xp.ndim == 1
        i = numpy.searchsorted(xp, x).clip(1, len(xp)-1)
        x0 = xp[i-1]
        x1 = xp[i]
        d = ((x-x0)/(x1-x0)).reshape(-1, 1)
        f0 = self[i-1]
        f1 = self[i]
        return (1-d)*f0+d*f1

    def fillna(self, value):
        self[numpy.isnan(self)] = value
        return self

    def unique(self, axis=-2) -> Vector:
        return numpy.unique(self, axis=axis).view(Vector)

    def to_pcd(self, path, fields=""):
        w = self.shape[-1]
        pcd = self.reshape(-1, w)
        width = len(pcd)
        size = " ".join(["4"] * w)
        tp = " ".join(["F"] * w)
        count = " ".join(["1"] * w)
        if not fields:
            fields = " ".join([str(i) for i in range(w)])
        f = open(path, "w")
        f.write('''# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS {}
SIZE {}
TYPE {}
COUNT {}
WIDTH {}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {}
DATA ascii
'''.format(fields, size, tp, count, width, width))
        for p in pcd:
            f.write(" ".join([str(a) for a in p]) + "\n")
        f.close()

    def to_csv(self, path):
        '''
        Save data to a csv file
        '''
        if hasattr(self, "columns"):
            header = ",".join(self.columns)
        else:
            header = ...
        numpy.savetxt(path, self, delimiter=",", header=header, comments="")

    def to_npy(self, path):
        numpy.save(path, self)

    def as_image(self, nonzero=True, sample_rate=None):
        '''
        Visualize the vector as an image, with mapped colors from black to yellow or the image's own colors
        '''
        if not sample_rate:
            sample_rate = max(round(self.size / 1e6), 1)
        sample = self[::-sample_rate, ::sample_rate]
        h, w, *_ = self.shape
        ret = Vector3.grid(range(0, w, sample_rate),
                           range(0, h, sample_rate)).as_point()
        if sample.dtype == numpy.uint8:
            color = Color(sample / 255)
        else:
            color = Color.map(sample)
        ret.color = color.transpose(1, 0, 2)
        if nonzero:
            return ret[ret.color.rgb.any(-1)]
        return ret


class Vector2(Vector):
    BASE_SHAPE = 2,

    def __new__(cls, data: list | numpy.ndarray = []):
        return super().__new__(cls, data)


class Vector3(Vector):
    BASE_SHAPE = 3,

    def __new__(cls, data: list | numpy.ndarray = [], x=0, y=0, z=0):
        '''
            Represent points, positions and translations
        '''
        if numpy.any(data):
            return super().__new__(cls, data)
        else:
            n = max(numpy.shape(x), numpy.shape(y), numpy.shape(z))
            ret = numpy.empty(n + cls.BASE_SHAPE).view(cls)
            ret.x = x
            ret.y = y
            ret.z = z
            return ret

    @classmethod
    def grid(cls, x=0, y=0, z=0) -> Vector3:
        n = numpy.shape(x) + numpy.shape(y) + numpy.shape(z)
        ret = numpy.empty(n+(3, )).view(Vector3)
        i = numpy.arange(len(n))
        ret.x = numpy.expand_dims(x, axis=i[i != 0].tolist())
        ret.y = numpy.expand_dims(y, axis=i[i != 1].tolist())
        ret.z = numpy.expand_dims(z, axis=i[i != 2].tolist())
        return ret

    @classmethod
    def circle(cls, radius=1, segments=20) -> Vector3:
        a = numpy.linspace(0, 2*numpy.pi, segments, False)
        return cls(x=radius * numpy.sin(a), y=radius * numpy.cos(a))

    def __matmul__(self, value: Transform) -> Vector3:
        if type(value) is Transform:
            return numpy.matmul(self.H, value[..., None, :, :])[..., 0:3].view(Vector3)
        else:
            return super().__matmul__(value)

    def dot(self, v) -> Vector:
        product = self * v
        return numpy.sum(product, axis=product.ndim - 1).view(Vector)

    def cross(self, v: numpy.ndarray) -> Vector3:
        return numpy.cross(self, v).view(self.__class__)

    def angle_to_vector(self, to: numpy.ndarray) -> Vector3:
        cos = self.dot(to) / self.L / Vector3(to).L
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
        return self.dot(v) / Vector3(v).L

    def vector_projection(self, v: numpy.ndarray) -> Vector3:
        s = self.scalar_projection(v) / Vector3(v).L
        return s.reshape(-1, 1) * v

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

    def closest_point_to_points(self, points: Vector3 | numpy.ndarray | list) -> Vector3:
        '''
        return closest point indexes of one point cloud to another point cloud, and also return indexes of the pair points in the another point cloud
        both self and points should be flattened
        '''
        pts = Vector3(points)
        assert self.ndim < 3, "self should be flattened"
        assert pts.ndim < 3, "parameter `points` should be flattened"
        d: Vector = (self[..., numpy.newaxis, :] - pts).L
        d = d.reshape(*d.shape[:-2], -1)
        idx = d.argmin(d.ndim-1)
        spts = sum(pts.n)
        idx0 = idx//spts
        idx1 = idx % spts
        return idx0, idx1

    def distance_to_points(self, points: Vector3) -> numpy.ndarray:
        return (self[..., None, :] - points).L.min(axis=-1).mean()

    def as_point(self, color=None, colormap=None) -> Point:
        entity = Point(*self.n).paint(color, colormap)
        entity.xyz = self
        return entity

    def as_line(self) -> LineSegment:
        n = list(self.n)
        n[-1] = (n[-1] - 1) * 2
        entity = LineSegment(*n)
        entity.start.xyz = self[..., :-1, :]
        entity.end.xyz = self[..., 1:, :]
        return entity

    def as_lineloop(self) -> LineSegment:
        n = list(self.n)
        n[-1] = n[-1] * 2
        entity = LineSegment(*n)
        entity.start.xyz = self
        entity.end.xyz = numpy.roll(self, -1, axis=self.ndim - 2)
        return entity

    def as_linesegment(self) -> LineSegment:
        entity = LineSegment(*self.n)
        entity.xyz = self
        return entity

    def as_shape(self) -> Triangle:
        v = numpy.repeat(self, 3, axis=self.ndim-2)
        v = numpy.roll(v, 1, axis=v.ndim-2)
        c = self.M[..., numpy.newaxis, :]
        v[..., 1::3, :] = c
        return v.view(Vector3).as_triangle()

    def as_triangle(self) -> Triangle:
        entity = Triangle(*self.n)
        entity.xyz = self
        return entity

    def as_vector(self) -> LineSegment:
        entity = LineSegment(*self.n, 2)
        entity.start.xyz = 0
        entity.end.xyz = numpy.expand_dims(self, axis=self.ndim - 1)
        return entity


class Vector4(Vector):
    BASE_SHAPE = 4,

    def __new__(cls, xyzw_list: list | numpy.ndarray = [], x=0, y=0, z=0, w=1):
        if numpy.any(xyzw_list):
            return super().__new__(cls, xyzw_list)
        else:
            n = max(numpy.shape(x), numpy.shape(y),
                    numpy.shape(z), numpy.shape(w))
            ret = numpy.empty(n + cls.BASE_SHAPE).view(cls)
            ret.x = x
            ret.y = y
            ret.z = z
            ret.w = w
            return ret

    @property
    def wxyz(self) -> Vector:
        ret = numpy.empty(self.n + self.BASE_SHAPE).view(Vector)
        ret[..., 0] = self.w
        ret[..., 1:4] = self.xyz
        return ret

    def from_axis_angle_to_quaternion(self) -> Vector4:
        q = numpy.empty(self.n + self.BASE_SHAPE).view(Vector4)
        q.xyz = numpy.sin(self.w / 2)[..., numpy.newaxis] * self.xyz.U
        q.w = numpy.cos(self.w / 2)
        return q

    def from_quaternion_to_axis_angle(self) -> Vector4:
        q = numpy.empty(self.n + self.BASE_SHAPE).view(Vector4)
        q.w = numpy.arccos(self.w) * 2
        sin_ha = numpy.sin(q.w / 2)[..., numpy.newaxis]
        q.xyz = numpy.divide(self.xyz, sin_ha,
                             where=sin_ha != 0)
        return q


class Transform(Vector):
    BASE_SHAPE = 4, 4

    def __new__(cls, data: list | numpy.ndarray = numpy.eye(4)):
        return super().__new__(cls, data)

    @classmethod
    def from_translation(cls, xyz_list: list | numpy.ndarray = [], x=0, y=0, z=0) -> Transform:
        '''
        translation matrix
        '''
        vec = Vector3(xyz_list, x, y, z)
        ret = Transform().tile(*vec.n)
        ret[..., 3, 0] = vec[..., 0]
        ret[..., 3, 1] = vec[..., 1]
        ret[..., 3, 2] = vec[..., 2]
        return ret

    @classmethod
    def from_scaling(cls, xyz_list: list | numpy.ndarray = [], x=1, y=1, z=1) -> Transform:
        '''
        scaling matrix
        '''
        vec = Vector3(xyz_list, x, y, z)
        ret = Transform().tile(*vec.n)
        ret[..., 0, 0] = vec[..., 0]
        ret[..., 1, 1] = vec[..., 1]
        ret[..., 2, 2] = vec[..., 2]
        return ret

    @classmethod
    def Rx(cls, a) -> Transform:
        a = numpy.array(a)
        ret = numpy.full(a.shape + (4, 4), numpy.eye(4))
        cos = numpy.cos(a)
        sin = numpy.sin(a)
        ret[..., 1, 1] = cos
        ret[..., 1, 2] = sin
        ret[..., 2, 1] = -sin
        ret[..., 2, 2] = cos
        return ret.view(cls)

    def rx(self, a) -> Transform:
        return self @ self.Rx(a)

    @classmethod
    def Ry(cls, a) -> Transform:
        a = numpy.array(a)
        ret = numpy.full(a.shape + (4, 4), numpy.eye(4))
        cos = numpy.cos(a)
        sin = numpy.sin(a)
        ret[..., 0, 0] = cos
        ret[..., 0, 2] = -sin
        ret[..., 2, 0] = sin
        ret[..., 2, 2] = cos
        return ret.view(cls)

    def ry(self, a) -> Transform:
        return self @ self.Ry(a)

    @classmethod
    def Rz(cls, a) -> Transform:
        a = numpy.array(a)
        ret = numpy.full(a.shape + (4, 4), numpy.eye(4))
        cos = numpy.cos(a)
        sin = numpy.sin(a)
        ret[..., 0, 0] = cos
        ret[..., 0, 1] = sin
        ret[..., 1, 0] = -sin
        ret[..., 1, 1] = cos
        return ret.view(cls)

    def rz(self, a) -> Transform:
        return self @ self.Rz(a)

    @classmethod
    def from_axis_angle(cls, xyz_angle_list: list | Vector4 = [], axis=[0, 0, 1], angle=0) -> Transform:
        axis = Vector3(axis)
        q = Vector4(xyz_angle_list, axis.x, axis.y, axis.z,
                    angle).from_axis_angle_to_quaternion()
        return cls.from_quaternion(q)

    def as_axis_angle(self):
        return self.as_quaternion().from_quaternion_to_axis_angle()

    @classmethod
    def from_rotation_vector(cls, xyz_list: list | Vector3 = [], x=0, y=0, z=0) -> Transform:
        rv = Vector3(xyz_list, x, y, z)
        axis_angle_list = Vector4().tile(*rv.n)
        axis_angle_list.w = rv.L
        axis_angle_list.xyz = rv.U
        return cls.from_axis_angle(axis_angle_list)

    def as_rotation_vector(self):
        q = self.as_axis_angle()
        return q.xyz.U * q.w[..., None]

    @classmethod
    def from_two_vectors(cls, a: list | Vector3, b: list | Vector3) -> Transform:
        a = Vector3(a)
        b = Vector3(b)
        q = Vector4().tile(*max(a.n, b.n))
        q.w = a.angle_to_vector(b).squeeze()
        q.xyz = a.cross(b)
        return cls.from_axis_angle(q)

    @classmethod
    def from_quaternion(cls, xyzw_list: list | numpy.ndarray) -> Transform:
        q = Vector4(xyzw_list)
        ret = Transform().tile(*q.shape[:-1])
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
        q = Vector4().tile(*self.n)
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
        ret = Vector3().tile(*self.n)
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
    def from_rpy(cls, angles_list: list | numpy.ndarray = [], r=0, p=0, y=0) -> Transform:
        return cls.from_euler('XYZ', Vector3(angles_list, r, p, y))

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
        self[:] = Transform.from_scaling(v) @ self.scaling.I @ self

    @property
    def rotation_vector(self) -> Vector3:
        return self.rotation.as_rotation_vector()

    @rotation_vector.setter
    def rotation_vector(self, v: Vector3):
        self.rotation = Transform.from_rotation_vector(v)

    @property
    def translation(self) -> Transform:
        ret = Transform().tile(*self.n)
        ret[..., 3, 0:3] = self.translation_vector
        return ret

    @translation.setter
    def translation(self, v: numpy.ndarray):
        self[..., 3, :3] = v[..., 3, :3]

    @property
    def scaling(self) -> Transform:
        ret = Transform().tile(*self.n)
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
    def T(self) -> Transform:
        return self.transpose(*range(self.ndim-2), -1, -2)

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

    def lerp(self, target_x, origin_x) -> Transform:
        '''
        Linear interpolation
        target_x: 1-D array, the series to be interpolated. For example, time series.
        origin_x: 1-D array, the series to interpolate 'target_x' into, with same length as self. 
        Only translation, rotation and scaling can be interpolated
        '''
        x = numpy.array(target_x)
        xp = numpy.array(origin_x)
        assert x.ndim <= xp.ndim == 1
        i = numpy.searchsorted(xp, x).clip(1, len(xp)-1)
        x0 = xp[i-1]
        x1 = xp[i]
        d = ((x-x0)/(x1-x0)).reshape(-1, 1)
        r0: Transform = self.rotation[i-1]
        r1: Transform = self.rotation[i]
        t0: Vector3 = self.translation_vector[i-1]
        t1: Vector3 = self.translation_vector[i]
        s0: Vector3 = self.scaling_vector[i-1]
        s1: Vector3 = self.scaling_vector[i]
        axis_angle = (r0.I@r1).as_axis_angle()
        rotation = r0 @ Transform.from_axis_angle(
            axis=axis_angle.xyz, angle=d.flatten()*axis_angle.w)
        translation = Transform.from_translation(
            (1-d)*t0+d*t1)
        scaling = Transform.from_scaling(
            (1-d)*s0+d*s1)
        return scaling@rotation@translation


class Color(Vector):
    BASE_SHAPE = 4,

    def __new__(cls, data: numpy.ndarray | list = [], r=0, g=0, b=0, a=1):
        data = numpy.array(data)
        if data.any():
            ret = super().__new__(cls, data)
            if data.shape[-1] < 4:
                ret.a = 1
        else:
            n = max(numpy.shape(r), numpy.shape(g),
                    numpy.shape(b), numpy.shape(a))
            ret = numpy.empty(n + cls.BASE_SHAPE)
            ret[..., 0] = r
            ret[..., 1] = g
            ret[..., 2] = b
            ret[..., 3] = a
        return ret.view(cls)

    @classmethod
    def map(cls, value: list | numpy.ndarray, start=None, end=None):
        '''
        Create a series of colors by giving a a series of value, from black to yellow
        '''
        value = numpy.array(value, numpy.float32)
        if start is None:
            start = numpy.amin(value, axis=tuple(range(value.ndim)))
        if end is None:
            end = numpy.amax(value, axis=tuple(range(value.ndim)))
        position = numpy.divide(value-start, end-start,
                                where=value != 0, out=numpy.zeros_like(value))
        if value.ndim <= 2:
            r = numpy.clip(position*1.67-0.67, 0, 1)
            g = numpy.clip(position*5-1, 0, 1)
            b = numpy.clip((0.2-abs(position-0.2))*5, 0, 1)
            return cls(r=r, g=g, b=b)
        else:
            ret = Color(position[..., :3])
            ret[..., 3] = 1
            return ret

    @classmethod
    def standard(cls, n):
        '''
        Create a standard color series with shape of (*n, 4)
        '''
        size = numpy.prod(n)
        c = int(numpy.power(size, 1/3)) + 1
        s = numpy.linspace(0.5, 1, c)
        rgb: Vector3 = Vector3.grid(x=s, y=s, z=s).flatten()[
            :size].reshape(n+(3,))
        return rgb.H.view(cls)

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

    @property
    def rgb(self):
        return self[..., :-1].view(Vector3)


class Point(Vector):
    BASE_SHAPE = 7,
    TYPE = "POINTS"

    def __new__(cls, *n):
        ret = numpy.empty(n + cls.BASE_SHAPE).view(cls)
        ret.color = Color.standard(n[:-1] + (1,))
        ret.color.a = 1
        return ret

    @property
    def color(self) -> Color:
        return self[..., 3:7].view(Color)

    @color.setter
    def color(self, v):
        self[..., 3:7] = v

    @property
    def normal(self):
        return self[..., 7:10].view(Vector3)

    @normal.setter
    def normal(self, v):
        self[..., 7:10] = v

    def paint(self, color=None, colormap=None):
        if colormap is not None:
            color = Color.map(colormap)
        elif color is None:
            color = Color.standard(self.n[:-1] + (1,))
        self.color = color
        return self

    def __add__(self, v: Point) -> Point:
        '''
        Concatenate two Point
        '''
        assert self.TYPE == v.TYPE, f"Different TYPE {self.TYPE}, {v.TYPE}"
        assert self.shape[1:] == v.shape[1:
                                         ], f"Different shape {self.shape[1:-1]}, {v.shape[1:-1]}"
        return numpy.concatenate((self, v), axis=0).view(Point)

    def __iadd__(self, v: Point) -> Point:
        self = self.__add__(v)
        return self

    def __matmul__(self, transform: Transform) -> Point:
        xyz = self.xyz @ transform
        ret = self.__class__(*xyz.n)
        ret.xyz = xyz
        ret.color = self.color
        return ret

    def _repr_html_(self):
        return render(self)._repr_html_()


class Triangle(Point):
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


def cube(size_x=1, size_y=1, size_z=1) -> LineSegment:
    k = Vector3.grid([-.5, .5], [-.5, .5], [-.5, .5]).flatten() * \
        (size_x, size_y, size_z)
    k: Vector3 = k[[0, 1, 2, 3, 4, 5, 6, 7, 0, 2,
                   2, 6, 6, 4, 4, 0, 1, 3, 3, 7, 7, 5, 5, 1], :]
    return k.as_linesegment()


def car(wheelbase=3, wheel_radius=0.3, track_width=1.6, height=1.5, front_overhang=1, rear_overhang=1) -> LineSegment:
    size_x = wheelbase+front_overhang+rear_overhang
    size_y = track_width
    size_z = height-wheel_radius
    body = cube(size_x, size_y, size_z).xyz
    body @= Transform.from_translation(x=size_x /
                                       2-rear_overhang, z=size_z/2+wheel_radius)
    wheel = Vector3.circle(wheel_radius).as_lineloop().xyz
    wheel @= Transform.from_rpy([pi/2, 0, 0]) @ Transform.from_translation(
        Vector3.grid(x=[wheelbase, 0], y=[-size_y/2, size_y/2], z=wheel_radius))
    return numpy.vstack((body.flatten(), wheel.flatten())).view(Vector3).as_linesegment()


def axis(size=5, dashed=False) -> LineSegment:
    dash_size = 40 if dashed else 2
    a: LineSegment = Vector3().tile(3, dash_size).as_linesegment()
    a[0].x = numpy.linspace(0, size, dash_size)
    a[1].y = numpy.linspace(0, size, dash_size)
    a[2].z = numpy.linspace(0, size, dash_size)
    a[0].color = Color(r=1)
    a[1].color = Color(g=1)
    a[2].color = Color(b=1)
    return a.flatten()


def camera(pixel_width, pixel_height, focal_length_in_pixels, pixel_size=1e-3):
    ret = LineSegment(16)
    ret.start[:4].xyz = 0
    corners = numpy.array([
        [pixel_width/2, pixel_height/2, focal_length_in_pixels],
        [-pixel_width/2, pixel_height/2, focal_length_in_pixels],
        [-pixel_width/2, -pixel_height/2, focal_length_in_pixels],
        [pixel_width/2, -pixel_height/2, focal_length_in_pixels]], dtype=numpy.float64) * pixel_size
    ret.end[:4].xyz = corners
    ret.start[4:].xyz = corners
    ret.end[4:].xyz = numpy.roll(corners, 1, 0)
    return ret

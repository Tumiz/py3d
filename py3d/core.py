# Copyright (c) Tumiz.
# Distributed under the terms of the GPL-3.0 License.
from __future__ import annotations
import PIL.Image
import numpy
from typing import Dict
import pathlib
import json
import struct
import multiprocessing
import http.server
import socket
import base64

pi = numpy.arccos(-1)
__module__ = __import__(__name__)
numpy.set_printoptions(linewidth=600, suppress=True)


def sign(v):
    return numpy.sign(v, where=v != 0, out=numpy.ones_like(v))


def launch_server(ip, port):
    server = http.server.HTTPServer(
        (ip, port), http.server.SimpleHTTPRequestHandler)
    server.serve_forever()


class KDTree:
    class KDNode:
        def __init__(self, K, parent=None):
            self.dim = (parent.dim+1) % K if parent else 0
            self.split = None
            self.left = None
            self.right = None
            self.idx = []
            self.val = []

        def __repr__(self) -> str:
            ret = ""
            nodes = [self]
            while nodes:
                tmp = []
                for node in nodes:
                    if node.split is None:
                        ret += f"{node.idx}"
                    else:
                        ret += f"{node.split}"
                        tmp += [node.left, node.right]
                    ret += " "
                ret += "\n"
                nodes = tmp
            return ret

    def __init__(self, data, leaf_size=60):
        shape = numpy.shape(data)
        self.K = shape[-1] if len(shape) else 1
        self.data = numpy.reshape(data, (-1, self.K))
        self.leaf_size = leaf_size
        self.tree = self.load(numpy.arange(len(data)), None)

    def load(self, idx, parent: None | KDNode):
        size = len(idx)
        if size == 0:
            node = None
        else:
            node = self.KDNode(self.K, parent)
            if size <= self.leaf_size:
                node.idx = idx
                node.val = self.data[idx]
            else:
                sidx = idx[numpy.argsort(self.data[idx, node.dim])]
                mid = size//2
                node.split = self.data[sidx[mid], node.dim]
                node.left = self.load(sidx[:mid], node)
                node.right = self.load(sidx[mid:], node)
        return node

    def search(self, v, node: KDNode, d=float("inf"), i=-1):
        if node.split is None:
            ds = numpy.linalg.norm(v - node.val, axis=1)
            li = numpy.argmin(ds)
            td = ds[li]
            if d - td > 0:
                d, i = td, node.idx[li]
        else:
            dd = node.split - v[node.dim]
            if dd > 0:
                d, i = self.search(v, node.left, d, i)
                if d - dd > 0:
                    d, i = self.search(v, node.right, d, i)
            else:
                d, i = self.search(v, node.right, d, i)
                if d + dd > 0:
                    d, i = self.search(v, node.left, d, i)
        return d, i

    def query(self, values):
        ds = []
        ids = []
        shape = numpy.shape(values)
        values = numpy.reshape(values, (-1, self.K))
        for v in values:
            d, i = self.search(v, self.tree)
            ds.append(d)
            ids.append(i)
        return numpy.reshape(ds, shape[:-1]), numpy.reshape(ids, shape[:-1])


class Viewer:
    try:
        __display__ = display
    except:
        __display__ = None

    class JS:
        def __init__(self, script) -> None:
            self.script = script

        def _repr_javascript_(self):
            return self.script

    class HTML:
        def __init__(self, script) -> None:
            self.script = script

        def _repr_html_(self):
            return self.script

    __preload__ = open(pathlib.Path(__file__).parent/"viewer.html").read()

    def __init__(self) -> None:
        self.cache: Dict[float, list] = {}
        self.min = []
        self.max = []
        self.viewpoint = None
        self.lookat = None
        self.up = None
        self.size = (600, 1000)
        self.handle = None

    def __render_args__(self, t, **args):
        t = round(t, 3)
        if t in self.cache:
            self.cache[t].append(args)
        else:
            self.cache[t] = [args]
        return self

    def _repr_html_(self):
        self.handle = None
        html = self.__preload__.replace("PY#D_ARGS", json.dumps(self.__dict__))
        self.cache.clear()
        self.max = []
        self.min = []
        self.viewpoint = None
        self.lookat = None
        self.up = None
        self.size = (600, 1000)
        return html

    def show(self, inplace=False, viewpoint=None, lookat=None, up=None, size=[600, 1000], in_jupyter=True, name="py3d", port=9871):
        '''
        same as py3d.show
        '''
        self.viewpoint = viewpoint
        self.lookat = lookat
        self.up = up
        self.size = size
        if in_jupyter and Viewer.__display__:
            if inplace:
                if self.handle:
                    self.handle.update(Viewer.JS(
                        f"viewer.toolbar.cache={self.cache};viewer.min={self.min};viewer.max={self.max};viewer.render()"))
                else:
                    Viewer.__display__(Viewer.HTML(self.__preload__.replace(
                        "PY#D_ARGS", json.dumps(self.__dict__))))
                    self.handle = Viewer.__display__(
                        Viewer.JS(""), display_id=f"{id(self)}")
                self.cache.clear()
            else:
                Viewer.__display__(self)
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
            if hasattr(obj, "texture") and obj.texture:
                ext = pathlib.Path(obj.texture).suffix
                texture = f"data:image/{ext};base64," + base64.b64encode(
                    open(obj.texture, "rb").read()).decode('utf-8')
            else:
                texture = ""
            return self.__render_args__(t=t,
                                        mode=obj.TYPE,
                                        vertex=obj.xyz.ravel().tolist(),
                                        color=obj.color.ravel().tolist(),
                                        normal=obj.normal.ravel().tolist(),
                                        pointsize=obj.pointsize if hasattr(
                                            obj, "pointsize") else 0,
                                        texture=texture)

    def label(self, text: str, position: list = [0, 0, 0], color="grey", t=0):
        if isinstance(color, Color):
            color = f"rgb({color.r*255} {color.g*255} {color.b*255})"
        return self.__render_args__(t=t, mode="TEXT", text=text,
                                    vertex=position, color=color)


viewer = Viewer()


def render(*objs, t=0):
    for obj in objs:
        viewer.render(obj, t)
    return viewer


def label(text, position: list = [0, 0, 0], color="grey", t=0):
    return viewer.label(text, position, color, t)


def show(inplace=False, viewpoint=None, lookat=None, up=None, size=[600, 1000], in_jupyter=True, name="py3d", port=9871):
    """
    Display all rendered objects in one scene

    Parameters
    ----------
    inplace: bool, default False
        If False, update the existing viewer, otherwise, display a new viewer.
    viewpoint : list[float] | None
        the position from where to view the scene.
    lookat: list[float] | None
        the position to look at
    up: list[float] | None
        up direction to view the scene
    inplace: bool
        update the output when displayed in jupyter
    size: (float, float)
        size of the viewer, (height, width)
    in_jupyter: bool
        display in jupyter, as a output, otherwise in a web browser
    name: str
        name of the page when displayed in a web browser
    port: int
        port to visit the page when displayed in a web browser

    Returns
    -------
    out: None
    """
    return viewer.show(inplace, viewpoint, lookat, up, size, in_jupyter, name, port)


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


def read_ply(path):
    return PLY(path)


def read_obj(obj_path, texture_path=""):
    return OBJ(obj_path, texture_path)


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


def chamfer_distance(A: Vector, B: Vector, f_score_threshold=None, return_distances=False, return_precisions=False) -> float:
    '''
    chamfer distance between two nd points

    Parameters
    ----------
    A, B: Vector
        Two n D points
    f_score_threshold: float | None, optional
        If defined, F-Score will be returned. The default is None
    return_distances: bool, optional
        If True, distances between each point and its nearest neighbor will be returned. The default is False
    return_precisions: bool, optional
        If True, precisions between two points will be returned. The default is False

    Returns
    -------
    chamfer_distance: float
        The average squared distance between pairs of nearest neighbors between A and B
    distances_from_A_to_B: numpy.ndarray, optional
        Distances between each point in A and its nearest neighbor in B, returned only when `return_distances` is True
    distances_from_B_to_A: numpy.ndarray, optional
        Distances between each point in B and its nearest neighbor in A, returned only when `return_distances` is True
    f_score: float, optional
        The F-Score, also known as the F-measure, returned only when `f_score_threshold` is defined as a number
    precision_from_A_to_B: numpy.ndarray, optional
        Precision from A to B
    precision_from_B_to_A: numpy.ndarray, optional
        Precision from B to A
    '''
    ret = []
    a = A.flatten()
    b = B.flatten()
    b2a, _ = KDTree(a).query(b)
    a2b, _ = KDTree(b).query(a)
    ret.append(((b2a**2).mean() + (a2b**2).mean()).item())
    if return_distances:
        ret += [a2b, b2a]
    if f_score_threshold:
        precision_a2b = (a2b < f_score_threshold).mean().item()
        precision_b2a = (b2a < f_score_threshold).mean().item()
        if precision_a2b + precision_b2a:
            f_score = 2 * precision_a2b * precision_b2a / \
                (precision_a2b + precision_b2a)
        else:
            f_score = 0
        ret.append(f_score)
        if return_precisions:
            ret += [precision_a2b, precision_b2a]
    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)


class PLY:
    def __init__(self, path=""):
        self.vertices = []
        self.faces = []
        if path:
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
            while n_vertex:
                if "ascii" in data_type:
                    c = f.readline().decode("utf-8").strip()
                    self.vertices.append([float(v) for v in c.split(" ")])
                else:
                    self.vertices.append([struct.unpack("f", f.read(4))[0]
                                          for i in range(3)])
                n_vertex -= 1
            while n_face:
                if "ascii" in data_type:
                    c = f.readline().decode("utf-8").strip()
                    size, *d = [int(v) for v in c.split(" ")]
                else:
                    size, = struct.unpack("B", f.read(1))
                    d = struct.unpack("<" + "i" * size, f.read(4 * size))
                self.faces.append(d)
                n_face -= 1

    def _repr_html_(self) -> str:
        if self.faces:
            return self.as_mesh()._repr_html_()
        else:
            return self.as_point()._repr_html_()

    def as_mesh(self) -> Triangle:
        tris = []
        for d in self.faces:
            s = len(d)
            for i in range(s-2):
                tris.append([d[0], d[i+1], d[i+2]])
        mesh = Triangle(self.vertices, base_shape=(10,))[tris].flatten()
        a = mesh.xyz[0::3, :]
        b = mesh.xyz[1::3, :]
        c = mesh.xyz[2::3, :]
        mesh.normal = (b - a).cross(c - b).U.repeat(3, axis=-2)
        mesh.color = Color.standard((1,))
        return mesh

    def as_point(self) -> Point:
        return Point(self.vertices)

    def save(self, path):
        f = open(path, "w")
        header = f"""\
ply
format ascii 1.0
element vertex {len(self.vertices)}
property float x
property float y
property float z
"""
        if self.faces:
            header += f"""
element face {len(self.faces)}
property list uchar int vertex_index
"""
        header += "end_header\n"
        f.write(header)
        for x, y, z in self.vertices:
            f.write(f"{x} {y} {z}\n")
        for ids in self.faces:
            f.write(f"{len(ids)} {' '.join([str(i) for i in ids])}\n")
        f.close()


class OBJ:
    '''
    class to parse obj file
    '''

    def __init__(self, obj_path, texture_path="") -> None:
        f = open(obj_path)
        v = []
        vt = []
        vi = []
        vti = []
        for l in f:
            tp, *data = l.split(" ")
            if tp == "v":
                v.append([float(e) for e in data])
            elif tp == "vt":
                vt.append([float(e) for e in data])
            elif tp == "f":
                if "/" in l:
                    vi.append([int(e.split("/")[0])-1 for e in data])
                    vti.append([int(e.split("/")[1])-1 for e in data])
                else:
                    vi.append([int(e)-1 for e in data])
        self.v = Vector(v)
        self.vt = Vector(vt)
        self.vi = Vector(vi)
        self.vti = Vector(vti)
        self.texture = texture_path

    def _repr_html_(self):
        if self.texture:
            return self.as_mesh()._repr_html_()
        elif any(self.vi):
            return self.as_wireframe()._repr_html_()
        else:
            return self.as_point()._repr_html_()

    def as_point(self, color=None, colormap=None, pointsize=2) -> Point:
        p = self.v.xyz.as_point(color, colormap, pointsize)
        if self.v.shape[-1] > 3:
            p.color = Color(self.v[..., 3:])
        return p

    def as_mesh(self) -> Triangle:
        m = self.v[self.vi].xyz.as_triangle()
        if self.texture:
            m.color = 0
            m.color.xy = self.vt[self.vti]
            m.texture = self.texture
        return m

    def as_wireframe(self) -> LineSegment:
        return self.v[self.vi].xyz.as_lineloop()

    def save(self, path):
        f = open(path, "w")
        for v in self.v:
            f.write(f"v {' '.join([str(e) for e in v])}\n")
        for v in self.vt:
            f.write(f"vt {' '.join([str(e) for e in v])}\n")
        for vi, vti in zip(self.vi, self.vti):
            row = "f"
            for a, b in zip(vi, vti):
                row += f" {a+1}/{b+1}"
            f.write(row + "\n")
        f.close()


class Vector(numpy.ndarray):
    '''
    Base class of Vector2, Vector3, Vector4 and Transform

    See https://tumiz.github.io/py3d/Vector.html for examples.
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

    def a(self, *keys) -> Vector:
        '''
        Get attributes by column names
        '''
        idx = [self.columns.index(key) for key in keys]
        return self[..., idx]

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
    def L1(self) -> Vector:
        '''
        L1 distance, alse known as Manhattan distance or Manhattan norm.
        '''
        return numpy.abs(self).sum(axis=-1)

    @property
    def L(self) -> Vector:
        '''
        L2 distance, also known as Euclidean distance or Euclidean norm.
        Same as `L2`
        '''
        return self.L2

    @property
    def L2s(self) -> Vector:
        '''
        Squared L2 distance
        '''
        return (self**2).sum(axis=-1)

    @property
    def L2(self) -> numpy.ndarray:
        '''
        L2 distance, also known as Euclidean distance or Euclidean norm.
        Same as `L`
        '''
        return numpy.sqrt(self.L2s)

    def diff(self, n=1) -> Vector:
        return numpy.diff(self, n, axis=self.ndim-2)

    def lerp(self, target_x, origin_x) -> Vector:
        '''
        Linear interpolation, only translation, rotation and scaling can be interpolated

        Parameters
        ----------
        target_x: 1-D array
            The series to be interpolated. For example, time series.
        origin_x: 1-D array
            The series to interpolate 'target_x' into, with same length as self. 

        Returns
        -------
        Interpolated values
        '''
        x = numpy.array(target_x)
        xp = numpy.array(origin_x)
        assert x.ndim <= xp.ndim == 1
        i = numpy.searchsorted(xp, x).clip(1, len(xp)-1)
        x0 = xp[i-1]
        x1 = xp[i]
        shape = (-1, 1) if self.ndim > 1 else (-1,)
        d = ((x-x0)/(x1-x0)).reshape(*shape)
        f0 = self[i-1]
        f1 = self[i]
        return (1-d)*f0+d*f1

    def fillna(self, value):
        self[numpy.isnan(self)] = value
        return self

    def dropna(self, axis=None):
        '''
        Drop nan elements along an axis, default axis is None
        '''
        if axis is None:
            return self[~numpy.isnan(self)]
        else:
            others = [i for i in range(self.ndim) if i != axis]
            c = self.transpose(axis, *others).reshape(self.shape[axis], -1)
            idx = numpy.where(~numpy.isnan(c).any(1))[0]
            return self.take(idx, axis)

    def unique(self, axis=-2) -> Vector:
        return numpy.unique(self, axis=axis).view(Vector)

    def split(self, indices_or_sections, axis=-1) -> tuple:
        '''
        Split a Vector into multiple sub-Vectors
        '''
        return numpy.split(self, indices_or_sections, axis)

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

    def to_ply(self, path):
        ply = PLY()
        ply.vertices = self.xyz
        ply.save(path)

    def to_csv(self, path, fmt="%.7f"):
        '''
        Save data to a csv file
        '''
        if hasattr(self, "columns"):
            header = ",".join(self.columns)
        else:
            header = ...
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        numpy.savetxt(path, self, fmt, ",", header=header, comments="")

    def to_txt(self, path, delimiter=" ", fmt="%.7f", **arg):
        '''
        Convert to text file
        '''
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        numpy.savetxt(path, self, fmt, delimiter, **arg)

    def to_npy(self, path):
        numpy.save(path, self)

    def to_image(self, path):
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = self
        if data.dtype != numpy.uint8:
            vmax = data.max()
            vmin = data.min()
            if vmax > 1 or vmin < 0:
                data = Color.map(data)
            data = (data*255).astype(numpy.uint8)
        if data.ndim == 2:
            data = data[..., None].repeat(3, -1)
        mode = "RGBA" if data.shape[-1] > 3 else "RGB"
        PIL.Image.fromarray(data, mode).save(path)

    def as_image(self):
        '''
        Visualize the vector as an image, with mapped colors from black to yellow or the image's own colors
        '''
        self.to_image(".py3d/texture.png")
        h, w, *_ = self.shape
        m = Triangle([
            [0, 0, 0, 0, 1, 0, 0],
            [w, 0, 0, 1, 1, 0, 0],
            [w, h, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [w, h, 0, 1, 0, 0, 0],
            [0, h, 0, 0, 0, 0, 0]
        ], texture=".py3d/texture.png")
        viewer.viewpoint = [w/2, h/2, -w]
        viewer.lookat = [w/2, h/2, 0]
        viewer.up = [0, -1, 0]
        return m


class Vector2(Vector):
    BASE_SHAPE = 2,

    def __new__(cls, data: list | numpy.ndarray = []):
        return super().__new__(cls, data)


class Vector3(Vector):
    '''
    3D vectors

    This class provides an interface to represent points, positions and translations. It is also used to represent scaling vectors and rotation vectors.

    See https://tumiz.github.io/py3d/Vector3.html for examples.
    '''
    BASE_SHAPE = 3,

    def __new__(cls, data: list | numpy.ndarray = [], x=0, y=0, z=0):
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
            return numpy.matmul(self.H, value)[..., 0:3].view(Vector3)
        else:
            return super().__matmul__(value)

    def dot(self, v) -> Vector:
        product = self * v
        return numpy.sum(product, axis=product.ndim - 1).view(Vector)

    def cross(self, v: numpy.ndarray) -> Vector3:
        return numpy.cross(self, v).view(self.__class__)

    def angle_to_vector(self, to: numpy.ndarray) -> Vector3:
        cos = self.dot(to) / self.L2 / Vector3(to).L2
        return numpy.arccos(cos)

    def angle_to_plane(self, normal: numpy.ndarray) -> float:
        return numpy.pi / 2 - self.angle_to_vector(normal)

    def scalar_projection(self, v: numpy.ndarray) -> float:
        return self.dot(v) / Vector3(v).L2

    def vector_projection(self, v: numpy.ndarray) -> Vector3:
        s = self.scalar_projection(v) / Vector3(v).L2
        return s.reshape(-1, 1) * v

    def distance_to_points(self, points: Vector3) -> numpy.ndarray:
        return (self[..., None, :] - points).L.min(axis=-1).mean()

    def as_point(self, color=None, colormap=None, pointsize=2, fillna_val=0) -> Point:
        entity = Point(self.fillna(fillna_val)).paint(
            color, colormap, pointsize)
        return entity

    def as_line(self) -> LineSegment:
        n = list(self.n)
        n[-1] = (n[-1] - 1) * 2
        entity = LineSegment().tile(*n)
        entity.start.xyz = self[..., :-1, :]
        entity.end.xyz = self[..., 1:, :]
        return entity

    def as_lineloop(self) -> LineSegment:
        n = list(self.n)
        n[-1] = n[-1] * 2
        entity = LineSegment().tile(*n)
        entity.start.xyz = self
        entity.end.xyz = numpy.roll(self, -1, axis=self.ndim - 2)
        return entity

    def as_linesegment(self) -> LineSegment:
        entity = LineSegment(self)
        return entity

    def as_shape(self) -> Triangle:
        v = numpy.repeat(self, 3, axis=self.ndim-2)
        v = numpy.roll(v, 1, axis=v.ndim-2)
        c = self.M[..., numpy.newaxis, :]
        v[..., 1::3, :] = c
        return v.view(Vector3).as_triangle()

    def as_triangle(self) -> Triangle:
        entity = Triangle(self)
        return entity

    def as_vector(self) -> LineSegment:
        entity = LineSegment().tile(*self.n, 2)
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
    '''
    4x4 matrix. 
    This class provides an interface to handle rotation, translation and scaling.
    See https://tumiz.github.io/py3d/Transform.html for examples.
    '''
    BASE_SHAPE = 4, 4

    def __new__(cls, data: list | numpy.ndarray = numpy.eye(4)):
        return super().__new__(cls, data)

    @classmethod
    def from_translation(cls, xyz_list: list | numpy.ndarray = [], x=0, y=0, z=0) -> Transform:
        '''
        Initialize a `Transform` from translation vectors

        Parameters
        ----------
        xyz_list: list | numpy.ndarray
            translation vectors
        x, y, z: float or array_like
            first, second and third elements of translation vectors

        Returns
        -------
        transform: Transform
            4x4 translation matrix
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
        Initialize a `Transform` from scaling vectors

        Parameters
        ----------
        xyz_list: list | numpy.ndarray
            scaling vectors
        x, y, z: float or array_like
            first, second and third elements of scaling vectors

        Returns
        -------
        transform: Transform
            4x4 scaling matrix
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
        '''
        Initialize a `Transform` from axis-angles

        Parameters
        ----------
        xyz_angle_list: array_like
            rotation axises and angles
        axis: array_like
            rotation axises
        angle: float or array_like
            rotation angle around the axis

        Returns
        -------
        transform: Transform
            4x4 rotation matrix
        '''
        axis = Vector3(axis)
        q = Vector4(xyz_angle_list, axis.x, axis.y, axis.z,
                    angle).from_axis_angle_to_quaternion()
        return cls.from_quaternion(q)

    def as_axis_angle(self):
        return self.as_quaternion().from_quaternion_to_axis_angle()

    @classmethod
    def from_rotation_vector(cls, xyz_list: list | Vector3 = [], x=0, y=0, z=0) -> Transform:
        '''
        Initialize a `Transform` from rotation vectors

        Parameters
        ----------
        xyz_list: list | numpy.ndarray
            rotation vectors
        x, y, z: float or array_like
            first, second and third elements of rotation vectors

        Returns
        -------
        transform: Transform
            4x4 rotation matrix
        '''
        rv = Vector3(xyz_list, x, y, z)
        axis_angle_list = Vector4().tile(*rv.n)
        axis_angle_list.w = rv.L2
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
        '''
        Initialize a 4x4 rotation matrix from quaternions

        Parameters
        ----------
        xyzw_list: list | numpy.ndarray
            quaternions

        Returns
        -------
        transform: Transform
            4x4 rotation matrix
        '''
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
        '''
        Initialize a 4x4 rotation matrix from euler angles

        Parameters
        ----------
        sequence: str
            Specifies sequence of axes for rotations. 
            Up to 3 characters belonging to the set {‘X’, ‘Y’, ‘Z’} for intrinsic rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations. 
            Extrinsic and intrinsic rotations cannot be mixed in one function call.
        angles_list: list | numpy.ndarray
            Euler angles specified in radians

        Returns
        -------
        transform: Transform
            4x4 rotation matrix
        '''
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
        return self[..., :3].view(Vector3)

    @rgb.setter
    def rgb(self, v):
        self[..., :3] = v


class Point(Vector):
    TYPE = "POINTS"

    def __new__(cls, data=[], pointsize=2, texture="", base_shape=(7,)):
        cls.BASE_SHAPE = base_shape
        ret = super().__new__(cls, data)
        if numpy.any(data) and numpy.shape(data)[-1] < 7 and not texture:
            ret.color = Color.standard(ret.shape[:-2] + (1,))
            ret.color.a = 1
        ret.pointsize = pointsize
        ret.texture = texture
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

    def paint(self, color=None, colormap=None, pointsize=0):
        if colormap is not None:
            color = Color.map(colormap)
        elif color is None:
            color = Color.standard(self.n[:-1] + (1,))
        self.color = color
        self.pointsize = pointsize
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
        ret = self.__class__(xyz)
        ret.color = self.color
        if hasattr(self, "texture"):
            ret.texture = self.texture
        return ret

    def _repr_html_(self):
        return viewer.render(self)._repr_html_()


class Triangle(Point):
    TYPE = "TRIANGLES"

    def __new__(cls, data=[], texture="", base_shape=(7,)):
        ret = super().__new__(cls, data, 0, texture, base_shape)
        return ret


class LineSegment(Point):
    TYPE = "LINES"

    def __new__(cls, data=[]):
        ret = super().__new__(cls, data, 0)
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
    ret = LineSegment().tile(16)
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

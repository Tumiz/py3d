# coding: utf-8
# Copyright (c) Tumiz.
# Distributed under the terms of the GPL-3.0 License.

from .server import *
from .transform import *
from abc import abstractmethod
from time import sleep

class Scenario(Source):
    def __init__(self, name, *objs):
        Source.__init__(self, name)
        self.t = 0
        self.objects = dict()
        self.add(*objs)
        self.render_log = dict()

    def add(self, *objs):
        for obj in objs:
            self.objects[obj.id] = obj

    def remove(self, *objs):
        for obj in objs:
            del self.objects[obj.id]

    def clear(self):
        self.t = 0
        self.objects = dict()
        self.render_log.clear()

    def step(self, dt=0.01):
        while self.paused:
            sleep(1)
        for oid in self.objects:
            self.objects[oid].step(dt)
        self.t += dt

    def info(self):
        ret = {"t": self.t, "paused": self.paused}
        tmp = {}
        for oid in self.objects:
            tmp.update(self.__info(self.objects[oid]))
        ret["objects"] = tmp
        return ret

    def __info(self, obj):
        ret = {obj.id: obj.info()}
        for child in obj.children:
            ret.update(self.__info(child))
        return ret

    def render(self, **log):
        self.render_log.update(log)
        info = self.info()
        info.update({"log": self.render_log})
        self.send_msg(info)


class Object3D(Transform):
    def __init__(self, name=None):
        Transform.__init__(self)
        if name:
            self.id = name
        else:
            self.id = id(self)
        self.cls = None
        self.color = Color.Rand()
        self.mass = 0
        self.__velocity = Vector3()
        self.__angular_velocity = Rotation3()
        self.__local_velocity = Vector3()
        self.__local_angular_velocity = Rotation3()

    @abstractmethod
    def on_step(self, dt):
        pass

    @property
    def velocity(self):
        return self.__velocity

    @velocity.setter
    def velocity(self, v):
        self.__velocity = v
        self.__local_velocity = Vector3()

    @property
    def local_velocity(self):
        return self.__local_velocity

    @local_velocity.setter
    def local_velocity(self, v):
        self.__local_velocity = v

    @property
    def angular_velocity(self):
        return self.__angular_velocity

    @angular_velocity.setter
    def angular_velocity(self, v):
        self.__angular_velocity = v
        self.__local_angular_velocity = Rotation3()

    @property
    def local_angular_velocity(self):
        return self.__local_angular_velocity

    @local_angular_velocity.setter
    def local_angular_velocity(self, v):
        self.__local_angular_velocity = v

    def step(self, dt):
        self.on_step(dt)
        if self.mass:
            self.__velocity = self.velocity+Vector3(0, 0, -9.8)*dt
        if self.__local_velocity != Vector3():
            self.__velocity = self.rotation*self.__local_velocity
            # print(self.__local_velocity)
        if self.__local_angular_velocity != Vector3():
            self.__angular_velocity = self.rotation * \
                self.__local_angular_velocity*self.rotation.I
            # print(self.__local_angular_velocity)
        if self.__velocity is not None:
            # print(self.position, self.__velocity)
            self.position = self.position+self.__velocity*dt
        else:
            raise Exception(self.id, "velocity is nan")
        if self.__angular_velocity is not None:
            self.rotation = self.__angular_velocity*dt*self.rotation
        else:
            raise Exception(self.id, "angular velocity is nan")
        print(self.angular_velocity)
        for child in self.children:
            child.step(dt)

    def info(self):
        ret = Transform.info(self)
        ret.update({"id": self.id, "class": self.cls,
                    "color": self.color.tolist()})
        return ret


class Color:
    def __new__(cls, r=0, g=0, b=0, a=1):
        return array([r, g, b, a])

    @staticmethod
    def Rand(a=1):
        r = random.rand(4)
        r[3] = a
        return r


class Cube(Object3D):
    def __init__(self, size_x=1, size_y=1, size_z=1, name=None):
        Object3D.__init__(self, name)
        self.cls = "Cube"
        self.scale = Vector3(size_x, size_y, size_z)


class Sphere(Object3D):
    def __init__(self, r=1):
        Object3D.__init__(self)
        self.cls = "Sphere"
        self.radius = r

    @property
    def radius(self):
        return self.scale[0]

    @radius.setter
    def radius(self, r):
        self.scale = Vector3(r, r, r)

    def collision_with(self, obj):
        if isinstance(obj, Sphere):
            norm = (obj.position-self.position).norm()
            # print(norm, obj.radius, self.radius)
            if norm < obj.radius+self.radius:
                return True
            else:
                return False
        return False


class XYZ(Object3D):
    def __init__(self):
        Object3D.__init__(self)
        self.line_width = 2
        self.cls = "XYZ"
        self.size = 3

    def info(self):
        ret = Object3D.info(self)
        ret['line_width'] = self.line_width
        ret["size"] = self.size
        return ret


class Line(Object3D):
    Type_Default = "Default"
    Type_Vector = "Vector"

    def __init__(self):
        Object3D.__init__(self)
        self.cls = "Line"
        self.points = empty((0, 3))
        self.width = 2
        self.type = Line.Type_Default

    def add_point(self, *argv):
        for a in argv:
            if isinstance(a, ndarray):
                self.points=append(self.points, a.reshape(1,3), axis=0)
            else:
                raise Exception(type(a), "is not acceptable")

    def info(self):
        ret = Object3D.info(self)
        ret['points'] = self.points.tolist()
        ret['line_width'] = self.width
        ret['type'] = self.type
        return ret


class Cylinder(Object3D):
    def __init__(self):
        Object3D.__init__(self)
        self.cls = "Cylinder"
        self.top_radius = 1
        self.bottom_radius = 1
        self.height = 1

    def set_axis(self, axis):
        self.rotation = Rotation3.Direction_change(Vector3(0, 1, 0), axis)

    def info(self):
        ret = Object3D.info(self)
        ret['top_radius'] = self.top_radius
        ret['bottom_radius'] = self.bottom_radius
        ret['height'] = self.height
        return ret


class Pipe(Object3D):
    def __init__(self):
        Object3D.__init__(self)
        self.cls = "Pipe"
        self.cross = []
        self.path = []

    def info(self):
        ret = Object3D.info(self)
        ret["path"] = self.path
        ret["cross"] = self.cross
        return ret


class Plane(Cube):
    def __init__(self, point, norm):
        Cube.__init__(self, 10, 10, 0.01)
        self.rotation = Rotation3.Direction_change(Vector3(0, 0, 1), norm)
        self.position = point


class Point(Sphere):
    def __init__(self, position):
        Sphere.__init__(self, 0.1)
        self.position = position


class Vector(Line):
    def __init__(self, *argv):
        Line.__init__(self)
        if len(argv) == 1:
            self.add_point(Vector3(), argv[0])
        elif len(argv) == 2:
            self.add_point(argv[0], argv[1])
        else:
            raise Exception("only 1 or 2 arguments are accepted")
        self.type = Line.Type_Vector

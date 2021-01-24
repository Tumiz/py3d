from numpy import *
from math import sin, cos, cos, atan2, acos, sqrt


class Vector3(ndarray):
    def __new__(cls, x=0, y=0, z=0):
        if isinstance(x, list) or isinstance(x, tuple) or isinstance(x, ndarray):
            return ndarray.__new__(cls, (3), buffer=array(x, dtype=float))
        else:
            return ndarray.__new__(cls, (3), buffer=array([x, y, z], dtype=float))

    @staticmethod
    def Rand(x=[0, 0], y=[0, 0], z=[0, 0]):
        low = Vector3(x[0], y[0], z[0])
        up = Vector3(x[1], y[1], z[1])
        return low+random.rand(3)*(up-low)

    def length(self):  # norm
        return linalg.norm(self)

    def normalize(self):
        l = self.length()
        if(l):
            self.data = (self/l).data
        else:
            raise ValueError("Zero vector cant be normalized")

    def normalized(self):  # unit vector, direction vector
        l = self.length()
        if l:
            return self/l
        else:
            return None

    def cross(self, v):
        return Vector3(cross(self, v))

    def angle_to(self, to):
        cos = self.dot(to)/self.length()/to.length()
        return acos(cos)

    def rotation_to(self, to):
        axis = self.cross(to)
        angle = self.angle_to(to)
        return axis, angle

    def perpendicular_to(self, to):
        return self.dot(to) == 0

    def distance_to(self, p0, p1):
        v0 = p1-p0
        v1 = self-p0
        return v0.cross(v1).length()/v0.length()

    def area(self, p0, p1=None):
        if p1 is None:
            return self.cross(p0).length()
        else:
            v0 = p1-p0
            v1 = self-p0
            return v0.cross(v1).length()

    def clone(self):
        return Vector3(self)

    def __eq__(self, v):
        return self.data == v.data

    def __ne__(self, v):
        return self.data != v.data


class Rotation3(ndarray):
    def __new__(cls, matrix=eye(3)):
        return ndarray.__new__(cls, (3, 3), buffer=array(matrix, dtype=float))

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
        axis_n = axis.length()
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
        angle = acos(before.dot(after)/before.length()/after.length())
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

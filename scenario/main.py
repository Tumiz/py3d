# coding: utf-8
from .server import *
from math import *
import torch

class Vector3: 
    def __new__(cls,x=0,y=0,z=0):
        return torch.Tensor([float(x),float(y),float(z)])

class Rotation:
    def __init__(self):
        self.matrix=torch.eye(4)
    @staticmethod
    def eular(x=0,y=0,z=0):
        ret=Rotation()
        ret.matrix=Rotation.rz_matrix(z).mm(Rotation.ry_matrix(y)).mm(Rotation.rx_matrix(x))
        return ret
    @staticmethod
    def quaternion(x,y,z,w):
        ret=Rotation()
        ret.matrix=torch.tensor([
            [2*(pow(x,2)+pow(w,2))-1,2*(x*y-w*z),2*(x*z+w*y),0],
            [2*(x*y+w*z),2*(pow(w,2)+pow(y,2))-1,2*(y*z-w*x),0],
            [2*(x*z-w*y),2*(y*z+w*x),2*(pow(w,2)+pow(z,2))-1,0],
            [0,0,0,1]
        ])
        return ret
    @staticmethod
    def axis_angle(axis,angle):
        axis=axis/axis.norm()
        w=cos(angle/2)
        x,y,z=sin(angle/2)*axis
        return Rotation.quaternion(x,y,z,w)
    @staticmethod
    def rx_matrix(x):
        return torch.tensor([
            [1, 0, 0,0],
            [0, cos(x), -sin(x),0],
            [0, sin(x), cos(x),0],
            [0,0,0,1]
        ])     
    @staticmethod
    def ry_matrix(y):
        return torch.tensor([
            [cos(y),0,sin(y),0],
            [0,1,0,0],
            [-sin(y),0,cos(y),0],
            [0,0,0,1]
        ])
    @staticmethod
    def rz_matrix(z):
        return torch.tensor([
            [cos(z),-sin(z),0,0],
            [sin(z),cos(z),0,0],
            [0,0,1,0],
            [0,0,0,1]
        ])
    @staticmethod
    def eular2quaternion(rx,ry,rz):
        sx,cx=sin(rx/2),cos(rx/2)
        sy,cy=sin(ry/2),cos(ry/2)
        sz,cz=sin(rz/2),cos(rz/2)
        w=cx*cy*cz+sx*sy*sz
        z=cx*cy*sz-sx*sy*cz
        y=cx*sy*cz+sx*cy*sz
        x=sx*cy*cz-cx*sy*sz
        return x,y,z,w
    def to_eular(self):
        y=atan2(-self.matrix[2,0],sqrt(pow(self.matrix[0,0],2)+pow(self.matrix[1,0],2)))
        z=atan2(self.matrix[1,0]/cos(y),self.matrix[0,0]/cos(y))
        x=atan2(self.matrix[2,1]/cos(y),self.matrix[2,2]/cos(y))
        return torch.tensor([x,y,z])
    def to_quaternion(self):
        w=0.5*sqrt(max(self.matrix[0,0]+self.matrix[1,1]+self.matrix[2,2]+1,0))
        x=(self.matrix[2,1]-self.matrix[1,2])/4/w
        y=(self.matrix[0,2]-self.matrix[2,0])/4/w
        z=(self.matrix[1,0]-self.matrix[0,1])/4/w
        return torch.tensor([x,y,z,w])
    def to_axis_angle(self):
        angle=acos((self.matrix[0,0]+self.matrix[1,1]+self.matrix[2,2]-1)/2)
        axis=torch.tensor([
            self.matrix[2,1]-self.matrix[1,2],
            self.matrix[0,2]-self.matrix[2,0],
            self.matrix[1,0]-self.matrix[0,1]
        ])
        return axis,angle
    def rotate_x(self,angle):
        self.matrix=Rotation.rx_matrix(angle).mm(self.matrix)
        return self
    def rotate_y(self,angle):
        self.matrix=Rotation.ry_matrix(angle).mm(self.matrix)
        return self
    def rotate_z(self,angle):
        self.matrix=Rotation.rz_matrix(angle).mm(self.matrix)
        return self
    def rotate_axis(self,axis,angle):
        self.matrix=Rotation.axis_angle(axis,angle).matrix.mm(self.matrix)
        return self
    def info(self):
        return self.to_quaternion().tolist()

class Transform:
    def __init__(self,position=Vector3(),rotation=Rotation(),scale=Vector3(1,1,1)):
        self.position=position
        self.rotation=rotation
        self.scale=scale
        
    def transform_position(self,loc):#local position
        tmp=torch.tensor([
            [loc[0]],
            [loc[1]],
            [loc[2]],
            [1]
            ])
        r=self.translation_matrix().mm(self.rotation.matrix).mm(self.scaling_matrix()).mm(tmp)[0:3,0]
        return r
    
    def transform_rotation(self,rot):#local rotation
        tmp=torch.tensor([
            [rot[0]]
        ])
        pass

    def translation_matrix(self):
        return torch.tensor([
            [1,0,0,self.position[0]],
            [0,1,0,self.position[1]],
            [0,0,1,self.position[2]],
            [0,0,0,1]
        ])

    def scaling_matrix(self):
        return torch.tensor([
            [self.scale[0],0,0,0],
            [0,self.scale[1],0,0],
            [0,0,self.scale[2],0],
            [0,0,0,1]
        ])
    
    def info(self):
        return {"position":self.position.tolist(),"rotation":self.rotation.info(),"scale":self.scale.tolist()}
    
    def __repr__(self):
        return json.dumps(self.info())
    
class Scenario:
    server=None
    def __init__(self):#single instance
        self.t=0
        self.objects=set()
        if Scenario.server is None:
            Scenario.server=Server()

    def add(self,*objs):
        for obj in objs:
            self.objects.add(obj)

    def remove(self,*objs):
        for obj in objs:
            self.objects.remove(obj)

    def step(self,dt=0.01):
        for obj in self.objects:
            obj.step(dt)
        self.t+=dt

    def info(self):
        ret=dict()
        for obj in self.objects:
            ret[obj.id]=obj.info()
        return ret
        
    def render(self):
        if not Scenario.server.is_alive():
            Scenario.server.start()
        Scenario.server.send_msg(self.info())
        
class Object3D(Transform):
    def __init__(self):
        Transform.__init__(self)
        self.id=id(self)
        self.type=self.__class__.__name__
        self.color=Color(1,1,1)
        self.mass=0
        self.velocity=None
        self.angular_velocity=None
        self.local_velocity=None
        self.local_angular_velocity=None
        self.children=set()

    def step(self,dt):
        if self.local_velocity is not None:
            self.position=self.transform_position(self.local_velocity*dt)
            # print('local v',self.position)
        elif self.velocity is not None:
            self.position+=self.velocity*dt
            # print('v',self.position)
        if self.local_angular_velocity is not None:
            self.rotation=self.transform_rotation(self.local_angular_velocity*dt)
            # print('local a',self.rotation,self.local_angular_velocity)
        elif self.angular_velocity is not None:
            self.rotation+=self.angular_velocity*dt
            # print('a',self.rotation,self.angular_velocity)
      
    def info(self):
        ret=Transform.info(self)
        ret.update({"id":self.id,"type":self.type,"color":self.color.tolist()})
        return ret
    
    def __repr__(self):
        return json.dumps(self.info())
    
class Color:
    def __new__(cls,r=0,g=0,b=0,a=1):
        return torch.tensor([float(r),float(g),float(b),float(a)])
        
class Cube(Object3D):
    def __init__(self):
        Object3D.__init__(self)

class Sphere(Object3D):
    def __init__(self):
        Object3D.__init__(self)

    @property
    def radius(self):
        return self.scale[0]

    @radius.setter
    def radius(self,r):
        self.scale=Vector3(r,r,r)

    def collision_with(self,obj):
        if isinstance(obj,Sphere):
            norm=(obj.position-self.position).norm()
            print(norm,obj.radius,self.radius)
            if norm<obj.radius+self.radius:
                return True
            else:
                return False
        return False

class XYZ(Object3D):
    def __init__(self):
        Object3D.__init__(self)
        self.line_width=1

    def info(self):
        ret=Object3D.info(self)
        ret['line_width']=self.line_width
        return ret

class Line(Object3D):
    def __init__(self):
        Object3D.__init__(self)
        self.points=[]
        self.line_width=1
        self.is_arrow=False

    def info(self):
        ret=Object3D.info(self)
        ret['points']=self.points
        ret['line_width']=self.line_width
        ret['is_arrow']=self.is_arrow
        return ret
                
class Cylinder(Object3D):
    def __init__(self):
        Object3D.__init__(self)
        self.top_radius=1
        self.bottom_radius=1
        self.height=1

    def info(self):
        ret=Object3D.info(self)
        ret['top_radius']=self.top_radius
        ret['bottom_radius']=self.bottom_radius
        ret['height']=self.height
        return ret
    
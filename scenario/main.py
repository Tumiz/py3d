# coding: utf-8
from torch import tensor,sin,cos,Tensor,ones
from .server import *

class Vector3: 
    def __new__(cls,x=0,y=0,z=0):
        return Tensor([float(x),float(y),float(z)])

class Transform:
    def __init__(self,position=Vector3(),rotation=Vector3(),scale=Vector3(1,1,1)):
        self.position=position
        self.rotation=rotation
        self.scale=scale
        
    def transform_position(self,loc):#local position
        tmp=tensor([
            [loc[0]],
            [loc[1]],
            [loc[2]],
            [1]
            ])
        r=self.translation_matrix().mm(self.rotation_x_matrix()).mm(self.rotation_y_matrix()).mm(self.rotation_z_matrix()).mm(self.scaling_matrix()).mm(tmp)[0:3,0]
        return r
    
    def transform_rotation(self,rot):#local rotation
        tmp=tensor([
            [rot[0]]
        ])
        pass

    def translation_matrix(self):
        return tensor([
            [1,0,0,self.position[0]],
            [0,1,0,self.position[1]],
            [0,0,1,self.position[2]],
            [0,0,0,1]
        ])

    def scaling_matrix(self):
        return tensor([
            [self.scale[0],0,0,0],
            [0,self.scale[1],0,0],
            [0,0,self.scale[2],0],
            [0,0,0,1]
        ])
    
    def rotation_x_matrix(self):
        x=self.rotation[0]
        return tensor([
            [1,0,0,0],
            [0,cos(x),-sin(x),0],
            [0,sin(x),cos(x),0],
            [0,0,0,1]
        ])

    def rotation_y_matrix(self):
        y=self.rotation[1]
        return tensor([
            [cos(y),0,sin(y),0],
            [0,1,0,0],
            [-sin(y),0,cos(y),0],
            [0,0,0,1]
        ])

    def rotation_z_matrix(self):
        z=self.rotation[2]
        return tensor([
            [cos(z),-sin(z),0,0],
            [sin(z),cos(z),0,0],
            [0,0,1,0],
            [0,0,0,1]
        ])
    
    def info(self):
        return {"position":self.position.tolist(),"rotation":self.rotation.tolist(),"scale":self.scale.tolist()}
    
    def __repr__(self):
        return json.dumps(self.info())
    
class Scenario(object):
    _instance=None
    def __init__(self):
        pass
    def __new__(cls):#single instance
        if cls._instance is None:
            cls._instance=object.__new__(cls)
            cls.t=0
            cls.objects=set()
            cls.server=Server()
        return cls._instance

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
        
    def render(self):
        if not self.server.is_alive():
            self.server.start()
            self.server.open_web()
        for obj in self.objects:
            self.server.send_msg(obj.info())
        
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
        return tensor([float(r),float(g),float(b),float(a)])
        
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
    
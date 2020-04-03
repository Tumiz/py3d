# coding: utf-8
from torch import tensor,sin,cos,Tensor,ones
from .server import *

class Vector3: 
    def __new__(cls,x=0,y=0,z=0):
        return Tensor([float(x),float(y),float(z)])

class Pose:
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
        
class Object3D:
    def __init__(self):
        self.id=id(self)
        self.type=""
        self.pose=Pose()
        self.color=Color()
        self.mass=0
        self.velocity=None
        self.angular_velocity=None
        self.local_velocity=None
        self.local_angular_velocity=None

    def step(self,dt):
        if self.local_velocity is not None:
            self.pose.position=self.pose.transform_position(self.local_velocity*dt)
            # print('local v',self.pose.position)
        elif self.velocity is not None:
            self.pose.position+=self.velocity*dt
            # print('v',self.pose.position)
        if self.local_angular_velocity is not None:
            self.pose.rotation=self.pose.transform_rotation(self.local_angular_velocity*dt)
            # print('local a',self.pose.rotation,self.local_angular_velocity)
        elif self.angular_velocity is not None:
            self.pose.rotation+=self.angular_velocity*dt
            # print('a',self.pose.rotation,self.angular_velocity)
      
    def info(self):
        return {"id":self.id,"class":self.type,"pose":self.pose.info(),"color":self.color.__dict__}
    
    def __repr__(self):
        return json.dumps(self.info())
    
class Color:
    def __init__(self,r=1,g=1,b=1,a=1):
        if isinstance(r, dict):
            self.r=r["r"]
            self.g=r["g"]
            self.b=r["b"]
            self.a=r["a"]
        elif isinstance(r,tuple) or isinstance(r,list):
            l=len(r)
            self.r=r[0]
            self.g=r[1] if l>=2 else 1
            self.b=r[2] if l>=3 else 1
            self.a=r[3] if l>=4 else 1
        else:
            self.r=r
            self.g=g
            self.b=b
            self.a=a

    def __repr__(self):
        return self.__class__.__name__+str(self.__dict__)
        
class Cube(Object3D):
    def __init__(self):
        Object3D.__init__(self)
        self.type="Cube"

class Sphere(Object3D):
    def __init__(self):
        Object3D.__init__(self)
        self.type="Sphere"

    @property
    def radius(self):
        return self.pose.scale[0]

    @radius.setter
    def radius(self,r):
        self.pose.scale=Vector3(r,r,r)

    def collision_with(self,obj):
        if isinstance(obj,Sphere):
            norm=(obj.pose.position-self.pose.position).norm()
            print(norm,obj.radius,self.radius)
            if norm<obj.radius+self.radius:
                return True
            else:
                return False
        return False

class XYZ(Object3D):
    def __init__(self):
        Object3D.__init__(self)
        self.type="XYZ"
        self.line_width=1

    def info(self):
        ret=Object3D.info(self)
        ret['linewidth']=self.line_width
        return ret

class Line(Object3D):
    def __init__(self):
        Object3D.__init__(self)
        self.points=[]
        self.width=1
                

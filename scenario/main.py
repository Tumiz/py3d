# coding: utf-8
import torch
from .webserver import *
def t_rx(theta):
    if not isinstance(theta,torch.Tensor):
        theta=torch.tensor(theta)
    return torch.tensor([
        [1,0,0],
        [0,torch.cos(theta),-torch.sin(theta)],
        [0,torch.sin(theta),torch.cos(theta)]
    ])
def t_ry(beta):
    if not isinstance(beta,torch.Tensor):
        beta=torch.tensor(beta)
    return torch.tensor([
        [torch.cos(beta),0,torch.sin(beta)],
        [0,1,0],
        [-torch.sin(beta),0,torch.cos(beta)]
    ])
def t_rz(gamma):
    if not isinstance(gamma,torch.Tensor):
        gamma=torch.tensor(gamma)
    return torch.tensor([
        [torch.cos(gamma),-torch.sin(gamma),0],
        [torch.sin(gamma),torch.cos(gamma),0],
        [0,0,1]
    ])

def t_r(rotation):#must be a tensor
    return t_rx(rotation[0]).mm(t_ry(rotation[1])).mm(t_rz(rotation[2]))

class Vector3: 
    def __new__(cls,x=0,y=0,z=0):
        return torch.Tensor([float(x),float(y),float(z)])

class Pose:
    def __init__(self,position=Vector3(),rotation=Vector3(),scale=Vector3(1,1,1)):
        self.position=position
        self.rotation=rotation
        self.scale=scale
        
    def transform(self,loc):#local position
        p=loc*self.scale
        r=t_r(self.rotation)
        p=r.mm(p.view(3,1)).view(3)
        p+=self.position
        print(loc,self.scale,self.position,self.rotation,r,p)
        return p
    
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
            cls.objects=dict()
            cls.ws=WebSocket()
        return cls._instance

    def add(self,obj):
        self.objects[obj.id]=obj

    def step(self,dt=0.01):
        for key,obj in self.objects.items():
            obj.pose.position+=obj.velocity*dt
            obj.pose.rotation+=obj.angular_velocity*dt
        self.t+=dt
        
    def render(self):
        for key,obj in self.objects.items():
            self.ws.send_msg(obj.info())
        
class Object3D:
    def __init__(self,name):
        self.id=name
        self.type=""
        self.pose=Pose()
        self.color=Color()
        self.mass=0
        self.velocity=Vector3()
        self.angular_velocity=Vector3()
        self.local_velocity=Vector3()
        self.local_angular_velocity=Vector3()
      
    def info(self):
        return {"id":self.id,"class":self.type,"pose":self.pose.info(),"color":self.color.__dict__}
    
    def __repr__(self):
        return json.dumps(self.info())
    
class Color:
    def __init__(self,r=0,g=0,b=0,a=1):
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
    def __init__(self,name):
        Object3D.__init__(self,name)
        self.type="Cube"

class Sphere(Object3D):
    def __init__(self,name):
        Object3D.__init__(self,name)
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

class Line(Object3D):
    def __init__(self,points=[]):
        Object3D.__init__(self)
        self._points=points
        self._width=1

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self,v):
        self._points=v
        self._publish(v)

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self,v):
        self._width=v
        self._publish(v)
                
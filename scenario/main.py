# coding: utf-8
# Copyright (c) Tumiz.
# Distributed under the terms of the GPL-3.0 License.

from .server import *
from math import *
from abc import abstractmethod
import torch
from time import sleep

def sign(x):
    if x>=0:
        return 1
    else:
        return -1

class Vector3(torch.Tensor):
    def __new__(cls,x=0,y=0,z=0):
        return torch.Tensor.__new__(cls)
    
    def __init__(self,x=0,y=0,z=0):
        if isinstance(x,tuple) or isinstance(x,list):
            x,y,z=x
        self.data=torch.tensor([x,y,z],dtype=torch.float32)
        
    def __eq__(self,v):
        return (self.data==v.data).sum().item()==3
    
    def __ne__(self,v):
        return (self.data==v.data).sum().item()!=3
        
    @staticmethod
    def Rand(x=[0,0],y=[0,0],z=[0,0]):
        low=torch.tensor([x[0],y[0],z[0]])
        up=torch.tensor([x[1],y[1],z[1]])
        return low+torch.rand(3)*(up-low)

class EularType:
    Extrinsic=0
    Intrinsic=1
    
class Rotation(torch.Tensor):

    def __init__(self,matrix=torch.eye(3)):
        self.data=matrix
   
    @staticmethod
    def Eular(x=0,y=0,z=0,t=EularType.Intrinsic):
        if t==EularType.Intrinsic:
            return Rotation.Rx(x)*Rotation.Ry(y)*Rotation.Rz(z)
        else:
            return Rotation.Rz(z)*Rotation.Ry(y)*Rotation.Rx(x)

    @staticmethod
    def Quaternion(x,y,z,w):
        return Rotation(torch.tensor([
            [2*(pow(x,2)+pow(w,2))-1,2*(x*y-w*z),2*(x*z+w*y)],
            [2*(x*y+w*z),2*(pow(w,2)+pow(y,2))-1,2*(y*z-w*x)],
            [2*(x*z-w*y),2*(y*z+w*x),2*(pow(w,2)+pow(z,2))-1]
        ]))

    @staticmethod
    def Axis_angle(axis,angle):
        axis_n=axis.norm()
        if axis_n:
            axis=axis/axis_n
            w=cos(angle/2)
            x,y,z=sin(angle/2)*axis
            return Rotation.Quaternion(x,y,z,w)
        else:
            return Rotation()
        
    @staticmethod
    def Direction_change(before,after):
        axis=before.cross(after)
        angle=acos(before.dot(after)/before.norm()/after.norm())
        return Rotation.Axis_angle(axis,angle)
    
    @staticmethod
    def Rx(a):
        return Rotation(torch.tensor([
            [1, 0, 0],
            [0, cos(a), -sin(a)],
            [0, sin(a), cos(a)]
        ]))
    
    @staticmethod
    def Ry(a):
        return Rotation(torch.tensor([
            [cos(a),0,sin(a)],
            [0,1,0],
            [-sin(a),0,cos(a)]
        ]))
    
    @staticmethod
    def Rz(a):
        return Rotation(torch.tensor([
            [cos(a),-sin(a),0],
            [sin(a),cos(a),0],
            [0,0,1]
        ]))
    
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
        
    def to_eular(self,t=EularType.Intrinsic):
        if t==EularType.Extrinsic:
            x=atan2(self[2,1],self[2,2])
            y=atan2(-self[2,0],sqrt(self[2,1]**2+self[2,2]**2))
            z=atan2(self[1,0],self[0,0])
        else:
            x=atan2(-self[1,2],self[2,2])
            y=atan2(self[0,2],sqrt(self[1,2]**2+self[2,2]**2))
            z=atan2(-self[0,1],self[0,0])
        return [x,y,z]
    
    def to_quaternion(self):
        w=0.5*sqrt(self[0,0]+self[1,1]+self[2,2]+1)
        x=0.5*sign(self[2,1]-self[1,2])*sqrt(max(0,self[0,0]-self[1,1]-self[2,2]+1))
        y=0.5*sign(self[0,2]-self[2,0])*sqrt(max(0,self[1,1]-self[2,2]-self[0,0]+1))
        z=0.5*sign(self[1,0]-self[0,1])*sqrt(max(0,self[2,2]-self[0,0]-self[1,1]+1))
        return [x,y,z,w]
    
    def to_axis_angle(self):
        angle=acos((self[0,0]+self[1,1]+self[2,2]-1)/2)
        axis=torch.tensor([
            self[2,1]-self[1,2],
            self[0,2]-self[2,0],
            self[1,0]-self[0,1]
        ])
        l=axis.norm()
        if l:
            return axis,angle
        else:
            return torch.tensor([0,0,1],dtype=torch.float32),0
    
    def rotate_x(self,angle):
        self.data=Rotation.Rx(angle).mm(self)
        return self
    
    def rotate_y(self,angle):
        self.data=Rotation.Ry(angle).mm(self)
        return self
    
    def rotate_z(self,angle):
        self.data=Rotation.Rz(angle).mm(self)
        return self
    
    def rotate_axis(self,axis,angle):
        self.data=Rotation.Axis_angle(axis,angle).mm(self)
        return self
    
    def __mul__(self,v):
        t=type(v)
        if t is float or t is int:
            axis,angle=self.to_axis_angle()
            angle*=v
            return Rotation.Axis_angle(axis,angle)
        elif t is Rotation:
            return Rotation(self.mm(v))
        elif v.dim()==1:
            return self.mv(v)
        else:
            return self.mm(v.permute(1,0)).permute(1,0)
        
    def __imul__(self,v):
        self=self*v
        return self

    def __eq__(self,r):
        return (self.data==r.data).sum().item()==16
    
    def __ne__(self,r):
        return (self.data==r.data).sum().item()!=16
    
    @property
    def I(self):
        return Rotation(self.inverse())
        
class Transform:
    def __init__(self):
        self.position=Vector3()
        self.rotation=Rotation()
        self.scale=Vector3(1,1,1)
        self.direction=Vector3(1,0,0)
        self.parent=None
        self.children=set()
        
    def add(self,*objs):
        for obj in objs:
            obj.parent=self
            self.children.add(obj)
            
    def world_position(self):
        parent=self.parent
        ret=self.position
        while parent:
            ret=parent.rotation*(parent.scale*ret)+parent.position
            parent=parent.parent
        return ret
    
    def world_rotation(self):
        parent=self.parent
        ret=self.rotation
        while parent:
            ret=parent.rotation*ret
            parent=parent.parent
        return ret
        
    def lookat(self,destination):
        self.rotation=Rotation.Direction_change(self.direction,destination-self.position)
    
    def info(self):
        return {"position":self.world_position().tolist(),"rotation":self.world_rotation().to_eular(),"scale":self.scale.tolist()}
    
class Scenario(Client):
    def __init__(self):
        Client.__init__(self)
        self.t=0
        self.objects=dict()

    def add(self,*objs):
        for obj in objs:
            self.objects[obj.id]=obj

    def remove(self,*objs):
        for obj in objs:
            del self.objects[obj.id]
            
    def reset(self):
        self.t=0
        self.objects=dict()

    def step(self,dt=0.01):
        while self.paused:
            sleep(1)
        for oid in self.objects:
            self.objects[oid].step(dt)
        self.t+=dt

    def info(self):
        ret={"t":self.t,"paused":self.paused}
        tmp={}
        for oid in self.objects:
            tmp.update(self.__info(self.objects[oid]))
        ret["objects"]=tmp
        return ret
    
    def __info(self,obj):
        ret={obj.id:obj.info()}
        for child in obj.children:
            ret.update(self.__info(child))
        return ret
        
    def render(self):
        self.send_msg(self.info())
        
class Object3D(Transform):
    def __init__(self,name=None):
        Transform.__init__(self)
        if name:
            self.id=name
        else:
            self.id=id(self)
        self.cls=None
        self.color=Color.Rand()
        self.mass=0
        self.__velocity=Vector3()
        self.__angular_velocity=Rotation()
        self.__local_velocity=Vector3()
        self.__local_angular_velocity=Rotation()

    @abstractmethod
    def on_step(self,dt):
        pass
    
    @property
    def velocity(self):
        return self.__velocity
    
    @velocity.setter
    def velocity(self,v):
        self.__velocity=v
        self.__local_velocity=Vector3()
    
    @property
    def local_velocity(self):
        return self.__local_velocity
    
    @local_velocity.setter
    def local_velocity(self,v):
        self.__local_velocity=v
        
    @property
    def angular_velocity(self):
        return self.__angular_velocity
    
    @angular_velocity.setter
    def angular_velocity(self,v):
        self.__angular_velocity=v
        self.__local_angular_velocity=Rotation()
        
    @property
    def local_angular_velocity(self):
        return self.__local_angular_velocity
    
    @local_angular_velocity.setter
    def local_angular_velocity(self,v):
        self.__local_angular_velocity=v
    
    def step(self,dt):
        self.on_step(dt)
        if self.mass:
            self.__velocity=self.velocity+Vector3(0,0,-9.8)*dt
        if self.__local_velocity != Vector3():
            self.__velocity=self.rotation*self.__local_velocity
            # print(self.__local_velocity)
        if self.__local_angular_velocity != Rotation():
            self.__angular_velocity=self.rotation*self.__local_angular_velocity*self.rotation.I
            # print(self.__local_angular_velocity)
        if self.__velocity is not None:
            self.position=self.position+self.__velocity*dt
        else:
            raise Exception(self.id,"velocity is nan")
#         print(self.id,self.velocity.tolist(),self.position.tolist(),self.angular_velocity.to_axis_angle())
        if self.__angular_velocity is not None:
            self.rotation=self.__angular_velocity*dt*self.rotation
        else:
            raise Exception(self.id,"angular velocity is nan")
#         print(self.id,self.angular_velocity.to_axis_angle(),self.rotation.to_axis_angle(),(self.angular_velocity*dt).to_axis_angle())
        for child in self.children:
            child.step(dt)
      
    def info(self):
        ret=Transform.info(self)
        ret.update({"id":self.id,"class":self.cls,"color":self.color.tolist()})
        return ret
    
class Color:
    def __new__(cls,r=0,g=0,b=0,a=1):
        return torch.tensor([r,g,b,a])
    @staticmethod
    def Rand(a=1):
        r=torch.rand(4)
        r[3]=a
        return r
    @staticmethod
    def White(a=1):
        return torch.tensor([1,1,1,a])
    @staticmethod
    def Black(a=1):
        return torch.tensor([0,0,0,a])
        
class Cube(Object3D):
    corners=torch.tensor([
        [-0.5000, -0.5000, -0.5000],
        [-0.5000, -0.5000,  0.5000],
        [-0.5000,  0.5000,  0.5000],
        [ 0.5000,  0.5000,  0.5000],
              [-0.5,0.5,-0.5],
              [0.5,-0.5,0.5],
              [0.5,-0.5,-0.5],
              [0.5,0.5,-0.5]
    ])
    def __init__(self,size_x=1,size_y=1,size_z=1):
        Object3D.__init__(self)
        self.cls="Cube"
        self.scale=Vector3(size_x,size_y,size_z)

class Sphere(Object3D):
    def __init__(self,r=1):
        Object3D.__init__(self)
        self.cls="Sphere"
        self.radius=r

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
        self.line_width=2
        self.cls="XYZ"
        self.color=Color.White()
        self.size=3

    def info(self):
        ret=Object3D.info(self)
        ret['line_width']=self.line_width
        ret["size"]=self.size
        return ret

class Line(Object3D):
    Type_Default="Default"
    Type_Vector="Vector"
    def __init__(self):
        Object3D.__init__(self)
        self.cls="Line"
        self.points=torch.empty(0,3)
        self.width=2
        self.type=Line.Type_Default
        
    @staticmethod    
    def Vector(*argv):
        ret=Line()
        if len(argv)==1:
            ret.add_point(Vector3(),argv[0])
        elif len(argv)==2:
            ret.add_point(argv[0],argv[1])
        else:
            raise Exception("not tensor")
        ret.type=Line.Type_Vector
        return ret
    
    def add_point(self,*argv):
        for a in argv:
            if isinstance(a,list) or isinstance(a,tuple):
                self.points=torch.cat((self.points,Vector3(a).unsqueeze(0)))
            elif isinstance(a,torch.Tensor):
                if a.size()==torch.Size([3]):
                    self.points=torch.cat((self.points,a.unsqueeze(0)))
                else:
                    self.points=torch.cat((self.points,a))
            else:
                raise Exception(type(a),"is not acceptable")

    def info(self):
        ret=Object3D.info(self)
        ret['points']=self.points.tolist()
        ret['line_width']=self.width
        ret['type']=self.type
        return ret
                
class Cylinder(Object3D):
    def __init__(self):
        Object3D.__init__(self)
        self.cls="Cylinder"
        self.top_radius=1
        self.bottom_radius=1
        self.height=1
        
    def set_axis(self,axis):
        self.rotation=Rotation.Direction_change(Vector3(0,1,0),axis)

    def info(self):
        ret=Object3D.info(self)
        ret['top_radius']=self.top_radius
        ret['bottom_radius']=self.bottom_radius
        ret['height']=self.height
        return ret
    
class Pipe(Object3D):
    def __init__(self):
        Object3D.__init__(self)
        self.cls="Pipe"
        self.cross=[]
        self.path=[]
        
    def info(self):
        ret=Object3D.info(self)
        ret["path"]=self.path
        ret["cross"]=self.cross
        return ret
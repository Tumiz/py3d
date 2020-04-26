# coding: utf-8
from .server import *
from math import *
from abc import abstractmethod
import torch
def sign(x):
    if x>=0:
        return 1
    else:
        return -1

class Vector3: 
    def __new__(cls,x=0,y=0,z=0):
        return torch.tensor([x,y,z],dtype=torch.float64)
    @staticmethod
    def rand(x=[0,0],y=[0,0],z=[0,0]):
        low=torch.tensor([x[0],y[0],z[0]],dtype=torch.float64)
        up=torch.tensor([x[1],y[1],z[1]],dtype=torch.float64)
        return low+torch.rand(3,dtype=torch.float64)*(up-low)

class Rotation:
    def __init__(self):
        self.matrix=torch.eye(4,dtype=torch.float64)
    @staticmethod
    def Eular(x=0,y=0,z=0):
        ret=Rotation()
        ret.matrix=Rotation.rz_matrix(z).mm(Rotation.ry_matrix(y)).mm(Rotation.rx_matrix(x))
        return ret
    @staticmethod
    def Quaternion(x,y,z,w):
        ret=Rotation()
        ret.matrix=torch.tensor([
            [2*(pow(x,2)+pow(w,2))-1,2*(x*y-w*z),2*(x*z+w*y),0],
            [2*(x*y+w*z),2*(pow(w,2)+pow(y,2))-1,2*(y*z-w*x),0],
            [2*(x*z-w*y),2*(y*z+w*x),2*(pow(w,2)+pow(z,2))-1,0],
            [0,0,0,1]
        ],dtype=torch.float64)
        return ret
    @staticmethod
    def Axis_angle(axis,angle):
        axis_n=axis.norm()
        if axis_n:
            axis=axis/axis.norm()
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
    def rx_matrix(x):
        return torch.tensor([
            [1, 0, 0,0],
            [0, cos(x), -sin(x),0],
            [0, sin(x), cos(x),0],
            [0,0,0,1]
        ],dtype=torch.float64)     
    @staticmethod
    def ry_matrix(y):
        return torch.tensor([
            [cos(y),0,sin(y),0],
            [0,1,0,0],
            [-sin(y),0,cos(y),0],
            [0,0,0,1]
        ],dtype=torch.float64)
    @staticmethod
    def rz_matrix(z):
        return torch.tensor([
            [cos(z),-sin(z),0,0],
            [sin(z),cos(z),0,0],
            [0,0,1,0],
            [0,0,0,1]
        ],dtype=torch.float64)
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
        return [x,y,z]
    def to_quaternion(self):
        w=0.5*sqrt(self.matrix[0,0]+self.matrix[1,1]+self.matrix[2,2]+1)
        x=0.5*sign(self.matrix[2,1]-self.matrix[1,2])*sqrt(max(0,self.matrix[0,0]-self.matrix[1,1]-self.matrix[2,2]+1))
        y=0.5*sign(self.matrix[0,2]-self.matrix[2,0])*sqrt(max(0,self.matrix[1,1]-self.matrix[2,2]-self.matrix[0,0]+1))
        z=0.5*sign(self.matrix[1,0]-self.matrix[0,1])*sqrt(max(0,self.matrix[2,2]-self.matrix[0,0]-self.matrix[1,1]+1))
        return [x,y,z,w]
    def to_axis_angle(self):
        angle=acos((self.matrix[0,0]+self.matrix[1,1]+self.matrix[2,2]-1)/2)
        axis=torch.tensor([
            self.matrix[2,1]-self.matrix[1,2],
            self.matrix[0,2]-self.matrix[2,0],
            self.matrix[1,0]-self.matrix[0,1]
        ],dtype=torch.float64)
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
        self.matrix=Rotation.Axis_angle(axis,angle).matrix.mm(self.matrix)
        return self
    def __mul__(self,v):
        if isinstance(v,float) or isinstance(v,int):
            axis,angle=self.to_axis_angle()
            angle*=v
            return Rotation.Axis_angle(axis,angle)
        elif isinstance(v,torch.Tensor):
            return self.matrix[0:3,0:3].mm(v.view(3,1)).view(3)
        elif isinstance(v,Rotation):
            self.matrix=v.matrix.mm(self.matrix)
            return self
        
    def __eq__(self,r):
        return (self.matrix==r.matrix).sum().item()==16
        
    def clone(self):
        ret=Rotation()
        ret.matrix=self.matrix.clone()
        return ret
        
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
            ret=parent.transform_local_position(ret)
            parent=parent.parent
        return ret
    
    def world_rotation(self):
        parent=self.parent
        ret=self.rotation.clone()
        while parent:
            ret*=parent.rotation
            parent=parent.parent
        return ret
        
    def lookat(self,destination):
        self.rotation=Rotation.Direction_change(self.direction,destination-self.position)
        
    def transform_local_position(self,loc):#local position
        tmp=torch.tensor([
            [loc[0]],
            [loc[1]],
            [loc[2]],
            [1]
            ])
        r=self.translation_matrix().mm(self.rotation.matrix).mm(self.scaling_matrix()).mm(tmp)[0:3,0]
        return r
    
    def transform_local_vector(self,vector):
        tmp=torch.tensor([
            [loc[0]],
            [loc[1]],
            [loc[2]],
            [1]
            ])
        r=mm(self.rotation.matrix).mm(self.scaling_matrix()).mm(tmp)[0:3,0]
        return r

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
        return {"position":self.world_position().tolist(),"rotation":self.world_rotation().to_quaternion(),"scale":self.scale.tolist()}
    
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
            ret.update(self.__info(obj))
        return ret
    
    def __info(self,obj):
        ret={obj.id:obj.info()}
        for child in obj.children:
            ret.update(self.__info(child))
        return ret
        
    def render(self):
        if not Scenario.server.is_alive():
            Scenario.server.start()
        Scenario.server.send_msg(self.info())
        
class Object3D(Transform):
    def __init__(self):
        Transform.__init__(self)
        self.id=id(self)
        self.cls=None
        self.color=Color(1,1,1)
        self.mass=0
        self.velocity=None
        self.angular_velocity=None
        self.local_velocity=None
        self.local_angular_velocity=None

    @abstractmethod
    def on_step(self):
        pass
    
    def step(self,dt):
        self.on_step()
        if self.local_velocity is not None:
            self.position=self.transform_local_position(self.local_velocity*dt)
#             print(self.id,'local v',self.position,self.local_velocity)
        elif self.velocity is not None:
            self.position+=self.velocity*dt
#             print(self.id,'v',self.position,self.local_velocity)
        if self.local_angular_velocity is not None:
            self.rotation*=self.local_angular_velocity*dt
#             print(self.id,'local a',self.rotation,self.local_angular_velocity)
        elif self.angular_velocity is not None:
            self.rotation*=(self.angular_velocity*dt)
#             print(self.id,'a',self.rotation,self.angular_velocity)
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
    def rand(a=1):
        r=torch.rand(4)
        r[3]=a
        return r
        
class Cube(Object3D):
    def __init__(self):
        Object3D.__init__(self)
        self.cls="Cube"

class Sphere(Object3D):
    def __init__(self):
        Object3D.__init__(self)
        self.cls="Sphere"

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
        self.size=3

    def info(self):
        ret=Object3D.info(self)
        ret['line_width']=self.line_width
        ret["size"]=self.size
        return ret

class Line(Object3D):
    def __init__(self):
        Object3D.__init__(self)
        self.cls="Line"
        self.points=[]
        self.width=2
        self.type="Default"
    @staticmethod    
    def Vector(*argv,color=None):
        ret=Line()
        if len(argv)==1:
            if isinstance(argv[0],list):
                ret.points=[[0,0,0],argv[0]]
            else:
                ret.points=[[0,0,0],argv[0].tolist()]
        elif len(argv)==2:
            ret.points=[argv[0],argv[1]]
        elif len(argv)==3:
            ret.points=[[0,0,0],[argv[0],argv[1],argv[2]]]
        else:
            return None
        ret.type="Vector"
        ret.color=color if color else Color.rand()
        return ret

    def info(self):
        ret=Object3D.info(self)
        ret['points']=self.points
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
    
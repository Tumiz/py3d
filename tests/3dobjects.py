from scenario import *
import time
import torch

scenario=Scenario()

a=Cube()
a.position=Vector3(2,3,-1)
a.scale=Vector3(5,5,0.1)
a.color=Color(r=1)

b=Sphere()
b.position=Vector3(-2,-2,2)
b.color=Color(g=1)

c=XYZ()
c.position=Vector3(1,1,1)
c.rotation=Vector3(0.2,0.1,0.3)
c.line_width=3
c.scale=Vector3(3,3,3)

d=Cylinder()
d.position=Vector3(3,3,3)
d.rotation=Vector3(0.2,0,0)
d.top_radius=1
d.bottom_radius=2
d.height=1
d.color=Color(b=1)

e=Line()
e.points=[[0,0,0],[0,0,9]]
e.line_width=2
e.color=Color(r=1,b=1)

scenario.add(a,b,c,d,e)
while True:
    d.color=torch.rand(4)
    a.position[2]=torch.rand(1)
    scenario.render()
    time.sleep(1)
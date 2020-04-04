from scenario import *
import time
scenario=Scenario()
c=Cube()
scenario.add(c)
c.position=Vector3(1,1,1)
c.rotation=Vector3(0.2,0.1,-0.6)
c.color=Color(g=1)
p=Cube()
p.position=c.transform_position(Vector3(0.5,0.5,0.5))
p.scale=Vector3(0.1,0.1,0.1)
p.color=Color(r=1)
scenario.add(p)
s=Sphere()
s.position=Vector3(0,1,1)
s.color=Color(b=1)
s.radius=0.4
scenario.add(s)
while True:
    scenario.render()
    time.sleep(0.1)
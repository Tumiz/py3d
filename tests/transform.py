from scenario import *
scenario=Scenario()
c=Cube()
scenario.add(c)
c.position=Vector3(1,1,1)
c.rotation=Vector3(0.2,0.1,-0.6)
c.scale=Vector3(2,1,3)
c.color=Color(g=1)
p=Cube()
p.position=c.transform_position(Vector3(0.5,0.5,0.5))
p.rotation=c.rotation
p.scale=Vector3(0.2,0.2,0.2)
p.color=Color(r=1)
scenario.add(p)

scenario.render()
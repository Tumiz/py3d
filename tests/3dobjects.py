from scenario import *
scenario=Scenario()
a=Cube()
a.pose.rotation=Vector3(0.1,0.4,0)
a.pose.position=Vector3(2,3,-1)
b=Sphere()
b.pose.position=Vector3(-2,-2,2)
c=XYZ()
c.pose.position=Vector3(1,1,1)
c.pose.rotation=Vector3(0.2,0.1,0.3)
c.line_width=2
c.pose.scale=Vector3(3,3,3)
scenario.add(a,b,c)
scenario.render()
from scenario import *
scenario=Scenario()
s=Sphere()
s.pose.position=Vector3(0,1,1)
s.color=Color(b=1)
s.radius=1
scenario.add(s)
s1=Sphere()
s1.pose.position=Vector3(1,1,1)
s1.color=Color(r=1)
s1.radius=0.5
scenario.add(s1)
print(s.collision_with(s1))
scenario.render()
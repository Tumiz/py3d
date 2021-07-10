from py3d import Vector3
from toweb import Space

points=Vector3.Rand(200)
p=Space("points")
p.render_points(points.tolist())
p1=Space("vectors")
p1.render_arrows(Vector3.Zeros(200).tolist(),points.tolist())
import toweb
p=toweb.Page("3d")
p.render_point(1,2,3)
p.render_arrow(1,2,3,4,5,6)
p.wait()
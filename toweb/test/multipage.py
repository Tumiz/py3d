from toweb import Chart,Log
c=Chart()
print(id(c.server),id(Chart.server),Chart.port,id(Chart.connections))
l=Log()
print(id(l.server),id(Log.server),Log.port,id(Log.connections))
# Copyright (c) Tumiz.
# Distributed under the terms of the GPL-3.0 License.

import os
import json
import time
import socket
import asyncio
import threading
import webbrowser
import tornado.web
import tornado.websocket
from IPython.display import IFrame, display

def address_in_use(port,ip='127.0.0.1'):
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    try:
        s.connect((ip,port))
        s.shutdown(2)
#         print('%s:%d is used' % (ip,port))
        return True
    except:
#         print('%s:%d is unused' % (ip,port))
        return False

class IndexHandler(tornado.web.RequestHandler):
    def get(self,v):
#         print("Index",v,self.request)
        self.render("viewer.html",port=Server.port,ID=v)

class StaticHandler(tornado.web.StaticFileHandler):
    def set_extra_headers(self, path):
        self.set_header("Cache-control", "no-cache")

class WSHandler(tornado.websocket.WebSocketHandler):       
    def open(self,v):
        if Server.connections.__contains__(v):
            self.server = Server.connections[v]["server"]
            self.server.clients.append(self)
            self.write_message(json.dumps(self.server.cache))
        # print("ws open",v,self.request,self.server.__dict__)
        
    def on_close(self):
#         print("ws close",self.request,self.close_code,self.close_reason)
        self.server.clients.remove(self)
    
def send_callback(handler,msg):
    handler.write_message(msg)

class Server:
    port=8000
    server=None
    connections=dict()
    loop=None
    def __new__(cls,name):
        if cls.server is None:
            cls.server=threading.Thread(target=Server.run)
            cls.server.setDaemon(True)
            while address_in_use(cls.port):
                cls.port+=1
            cls.server.start()
        if cls.connections.__contains__(name):
            return cls.connections[name]["server"]
        else:
            instance=super().__new__(cls)
            instance.name=name
            instance.clients=[]
            instance.cache=[]
            instance.url = "http://localhost:"+str(cls.port)+"/view/"+name
            cls.connections[name]=dict(server=instance,clients=instance.clients)
            return instance
        
    def send(self,msg):
        if len(self.clients) == 0:
            print("Please open", self.url)
            display(IFrame(src=self.url,width="100%",height="600px"))
        while len(self.clients) == 0:
            time.sleep(0.1)
        for client in self.clients:
            Server.loop.add_callback(send_callback,client,msg)

    def send_t(self,method,msg):
        cmd={"method":method,"time":time.time(),"data":msg}
        self.cache.append(cmd)
        self.send(json.dumps([cmd]))

    def clear(self):
        self.send_t("clear","")
        self.cache=[]

    def log(self,level,*msg):
        tmp=""
        for m in msg:
            tmp+=str(m)+" "
        self.send_t(level,tmp)

    def info(self,*msg):
        self.log("info",*msg)

    def err(self,*msg):
        self.log("err",*msg)

    def warn(self,*msg):
        self.log("warn",*msg)

    def plot(self,x,y=None):
        self.send_t("plot",{"x":x,"y":y})

    def wait(self):
        input("Press enter to exit")
        
    @staticmethod
    def run():
        loop=asyncio.new_event_loop()
        asyncio.set_event_loop(loop) #允许server在子线程中运�?
        static_path=os.path.join(os.path.dirname(__file__), "static")
        template_path=os.path.join(os.path.dirname(__file__), ".")
        app = tornado.web.Application([
                (r"/view/(.*)", IndexHandler),
                (r"/ws/(.*)", WSHandler),
                (r"/(.*)",StaticHandler)      
            ],
            static_path = static_path,
            template_path = template_path,
            debug = True,
            autoreload=True
        )
        http_server = tornado.httpserver.HTTPServer(app)
        http_server.listen(Server.port)
        Server.loop=tornado.ioloop.IOLoop.current()
        Server.loop.start() 

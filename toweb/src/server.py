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
            if self.server.cache:
                self.write_message(self.server.cache)
        # print("ws open",v,self.request,self.server.__dict__)
        
    def on_message(self, message):
#         print(message,self.server.paused)
        msg=json.loads(message)
        cmd=msg["cmd"]
        data=msg["data"]
        
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
    def __init__(self,name):
        if Server.server is None:
            Server.server=threading.Thread(target=Server.run)
            Server.server.setDaemon(True)
            while address_in_use(Server.port):
                Server.port+=1
            Server.server.start()
        if Server.connections.__contains__(name):
            self = Server.connections[name]["server"]
        else:
            self.name=name
            self.clients=[]
            self.cache=None
            self.url = "http://localhost:"+str(Server.port)+"/view/"+name
            Server.connections[name]=dict(server=self,clients=self.clients)
            print(self.url)
    
    def open(self):
        webbrowser.open(self.url,new=1)
        
    def send(self,msg):
        self.cache=msg
        while len(self.clients) == 0:
            time.sleep(0.1)
        for client in self.clients:
            Server.loop.add_callback(send_callback,client,msg)
        
    @staticmethod
    def run():
        loop=asyncio.new_event_loop()
        asyncio.set_event_loop(loop) #允许server在子线程中运行
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

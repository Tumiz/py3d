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
        self.render("viewer.html",port=Client.port,ID=v)

class WSHandler(tornado.websocket.WebSocketHandler):       
    def open(self,v):
        self.client=Client.clients[int(v)]
        if self.client.handler:
            self.client.handler.close()
        self.client.handler=self
        self.write_message(self.client.cache)
#         print("ws open",v,self.request,self.client.__dict__)
        
    def on_message(self, message):
#         print(message,self.client.paused)
        msg=json.loads(message)
        cmd=msg["cmd"]
        data=msg["data"]
        if cmd=="pause":
            self.client.paused=data=="⏹️"
        elif cmd=="key" and self.client.on_key:
            self.client.on_key(data)
        
    def on_close(self):
#         print("ws close",self.request,self.close_code,self.close_reason)
        self.client.handler=None
    
def send(handler,msg):
    handler.write_message(msg)

class Client:
    port=8000
    server=None
    clients=dict()
    loop=None
    def __init__(self):
        if Client.server is None:
            Client.server=threading.Thread(target=Client.run)
            Client.server.setDaemon(True)
            while address_in_use(Client.port):
                Client.port+=1
            Client.server.start()
        self.id=id(self)
        Client.clients[self.id]=self
        self.handler=None
        self.url="http://localhost:"+str(Client.port)+"/view/"+str(self.id)
        self.render_in_jupyter=True
        self.cache={}
        self.paused=False
        self.on_key=None
        
    def send_msg(self,msg):
#         print("handler",self.handler)
        if self.handler is None:
            if self.render_in_jupyter:
                display(IFrame(src=self.url,width="100%",height="600px"))
            else:
                webbrowser.open(self.url)
            while self.handler is None:
                time.sleep(0.1)
        Client.loop.add_callback(send,self.handler,msg)
        self.cache=msg
        
    @staticmethod
    def run():
        loop=asyncio.new_event_loop()
        asyncio.set_event_loop(loop) #允许server在子线程中运行
        static_path=os.path.join(os.path.dirname(__file__), "static")
        template_path=os.path.join(os.path.dirname(__file__), "template")
        app = tornado.web.Application([
                (r"/view/(.*)", IndexHandler),
                (r"/ws/(.*)", WSHandler),
                (r"/(.*)",tornado.web.StaticFileHandler)      
            ],
            static_path = static_path,
            template_path = template_path,
            debug = True
        )
        http_server = tornado.httpserver.HTTPServer(app)
        http_server.listen(Client.port)
        Client.loop=tornado.ioloop.IOLoop.current()
        Client.loop.start() 
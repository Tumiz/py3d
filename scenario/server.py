# Copyright (c) Tumiz.
# Distributed under the terms of the GPL-3.0 License.

import os
import time
import asyncio
import threading
import tornado.web
import tornado.websocket
import socket
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
#         print("ws open",v,self.request)
        self.id=int(v)
        Client.clients[self.id].handler=self
        
    def on_close(self):
#         print("ws close",self.request,self.close_code,self.close_reason)
        Client.clients[self.id].handler=None

class Client:
    port=8000
    server=None
    clients=dict()
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
        
    def send_msg(self,msg):
        if self.handler is None:
            display(IFrame(src="http://localhost:"+str(self.port)+"/view/"+str(self.id),width="100%",height="600px"))
            while self.handler is None:
                time.sleep(0.1)
        self.handler.write_message(msg)
        
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
        tornado.ioloop.IOLoop.current().start() 
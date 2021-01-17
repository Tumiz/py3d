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
        self.render("viewer.html",port=Source.port,ID=v)

class WSHandler(tornado.websocket.WebSocketHandler):       
    def open(self,v):
        if Source.connections.__contains__(v):
            self.source = Source.connections[v]["source"]
            self.source.sinks.append(self)
            self.write_message(self.source.cache)
        # print("ws open",v,self.request,self.source.__dict__)
        
    def on_message(self, message):
#         print(message,self.source.paused)
        msg=json.loads(message)
        cmd=msg["cmd"]
        data=msg["data"]
        if cmd=="pause":
            self.source.paused=data=="⏹️"
        elif cmd=="key" and self.source.on_key:
            self.source.on_key(data)
        
    def on_close(self):
#         print("ws close",self.request,self.close_code,self.close_reason)
        self.source.sinks.remove(self)
    
def send(handler,msg):
    handler.write_message(msg)

class Source:
    port=8000
    server=None
    connections=dict()
    loop=None
    def __init__(self,name):
        if Source.server is None:
            Source.server=threading.Thread(target=Source.run)
            Source.server.setDaemon(True)
            while address_in_use(Source.port):
                Source.port+=1
            Source.server.start()
        if Source.connections.__contains__(name):
            Source.connections[name]["source"]=self
            self.sinks = Source.connections[name]["sinks"]
        else:
            self.sinks=[]
            Source.connections[name]=dict(source=self,sinks=self.sinks)
        if len(self.sinks)==0:    
            print("open http://localhost:"+str(Source.port)+"/view/"+name)
        self.cache={}
        self.paused=False
        self.on_key=None
        
    def send_msg(self,msg):
        while len(self.sinks) == 0:
            time.sleep(0.1)
        for sink in self.sinks:
            Source.loop.add_callback(send,sink,msg)
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
        http_server.listen(Source.port)
        Source.loop=tornado.ioloop.IOLoop.current()
        Source.loop.start() 
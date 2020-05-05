# Copyright (c) Tumiz.
# Distributed under the terms of the GPL-3.0 License.

import os
import time
import asyncio
import threading
import tornado.web
import tornado.websocket
import webbrowser
import socket

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
    def get(self):
        if(self.request.uri=="/"):
            self.render("viewer.html",port=Server.port)

class WSHandler(tornado.websocket.WebSocketHandler):
    users=set()
    msg_cache={}
    def open(self):
#         print("open",id(self))
        self.users.add(self)
        self.write_message(WSHandler.msg_cache)

    def on_close(self):
#         print("close",id(self))
        self.users.remove(self)

    @staticmethod
    def send_msg(msg):
        WSHandler.msg_cache=msg
        users=WSHandler.users.copy()
        for u in users:
            if u.get_status()==101:
                u.write_message(msg)
#                 print(msg)

class Server(threading.Thread):
    port=8000
    def __init__(self):
        threading.Thread.__init__(self)
        self.setDaemon(True)
        while address_in_use(Server.port):
            Server.port+=1
        
    def send_msg(self,msg):
        t=0
        if len(WSHandler.users)==0:
            webbrowser.open("http://localhost:"+str(self.port),new=0,autoraise=False)
        while len(WSHandler.users) == 0:
            time.sleep(0.1)
            t+=0.1
        WSHandler.send_msg(msg)

    def run(self):
        loop=asyncio.new_event_loop()
        asyncio.set_event_loop(loop) #允许server在子线程中运行
        static_path=os.path.join(os.path.dirname(__file__), "static")
        template_path=os.path.join(os.path.dirname(__file__), "template")
        app = tornado.web.Application([
                (r"/", IndexHandler),
                (r"/ws", WSHandler),
                (r"/(.*)",tornado.web.StaticFileHandler)      
            ],
            static_path = static_path,
            template_path = template_path,
            debug = True
        )
        http_server = tornado.httpserver.HTTPServer(app)
        http_server.listen(Server.port)
        tornado.ioloop.IOLoop.current().start() 
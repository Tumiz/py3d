import os
import time
import asyncio
import threading
import tornado.web
import tornado.websocket
import webbrowser

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        if(self.request.uri=="/"):
            self.render("viewer.html")

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
    def __init__(self):
        threading.Thread.__init__(self)
        self.setDaemon(True)
        
    def open_web(self):
        webbrowser.open("http://localhost:8080/",new=0,autoraise=False)

    def send_msg(self,msg):
        t=0
        if len(WSHandler.users)==0:
            self.open_web()
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
        http_server.listen(8080)
        tornado.ioloop.IOLoop.current().start() 
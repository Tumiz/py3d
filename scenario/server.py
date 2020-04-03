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

class WebsocketHandler(tornado.websocket.WebSocketHandler):
    instance=None
    def __new__(cls, app, req, **kwargs):
        # print("__new__")
        if cls.instance is None:
            # print(app,req,kwargs)
            cls.instance=tornado.websocket.WebSocketHandler(app,req,**kwargs)
        return cls.instance

class Server(threading.Thread):
    def __init__(self):
        super().__init__()
        self.setDaemon(True)
        self.static_path=os.path.join(os.path.dirname(__file__), "static")
        self.template_path=os.path.join(os.path.dirname(__file__), "template")
        self.app = tornado.web.Application([
                (r"/", IndexHandler),
                (r"/ws", WebsocketHandler),
                (r"/(.*)",tornado.web.StaticFileHandler)      
            ],
            static_path = self.static_path,
            template_path = self.template_path,
            debug = True
            )
        
    def open_web(self):
        webbrowser.open("http://localhost:8080/",new=0,autoraise=False)

    def send_msg(self,msg):
        t=0
        while WebsocketHandler.instance is None:
            print("waiting for web to open, "+str(t)+" s\r",end='')
            time.sleep(0.1)
            t+=0.1
        WebsocketHandler.instance.write_message(msg)
        
    def run(self):
        # print("start")
        loop=asyncio.new_event_loop()
        asyncio.set_event_loop(loop) #允许server在子线程中运行
        http_server = tornado.httpserver.HTTPServer(self.app)
        http_server.listen(8080)
        tornado.ioloop.IOLoop.current().start() 
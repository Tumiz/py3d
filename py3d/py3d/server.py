# Copyright (c) Tumiz.
# Distributed under the terms of the GPL-3.0 License.
import os
import json
import time
import socket
import uuid
import asyncio
import threading
import inspect
import tornado.web
import tornado.websocket
from tornado import httputil
from typing import Any
from IPython.display import IFrame, display


def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

def address_in_use(port, ip="127.0.0.1"):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, port))
        s.shutdown(2)
        #         print('%s:%d is used' % (ip,port))
        return True
    except:
        # print('%s:%d is unused' % (ip,port))
        return False


class IndexHandler(tornado.web.RequestHandler):
    def get(self, v):
        #         print("Index",v,self.request)
        self.render("viewer.html", ID=v)


class StaticHandler(tornado.web.StaticFileHandler):
    def set_extra_headers(self, path):
        self.set_header("Cache-control", "no-cache")


class WSHandler(tornado.websocket.WebSocketHandler):
    def __init__(
        self,
        application: tornado.web.Application,
        request: httputil.HTTPServerRequest,
        **kwargs: Any
    ) -> None:
        super().__init__(application, request, **kwargs)
        self.server = None

    def open(self, v):
        # v is string always
        if v in Page.connections:
            self.server = Page.connections[v]
            self.server.clients.append(self)
            self.write_message(json.dumps(self.server.cache))
#         print("ws open",v,self.request,self.server.clients)

    def on_close(self):
        if self.server:
            self.server.clients.remove(self)


def send_callback(handler, msg):
    handler.write_message(msg)


def log(*msg):
    cur = inspect.currentframe()
    info = cur.f_back.f_back
    ret = info.f_code.co_filename + ":" + str(info.f_lineno) + "]"
    for m in msg:
        ret += str(m) + " "
    return ret


class Page:
    host = get_ip()
    port = 8000
    server = None
    connections = dict()
    __loop = None

    def __new__(cls, name=""):
        name = name if name else str(uuid.uuid1())
        if Page.server is None:
            while address_in_use(Page.port):
                Page.port += 1
            Page.server = threading.Thread(target=Page.run)
            Page.server.setDaemon(True)
            Page.server.start()
        if name in cls.connections:
            instance = cls.connections[name]
            return instance
        else:
            instance = super().__new__(cls)
            instance.name = name
            instance.clients = []
            instance.cache = []
            instance.url = "http://{}:" + str(cls.port) + "/view/" + name
            instance.iframe = IFrame(src=instance.url.format("127.0.0.1"), width="100%", height="600px")
            print("click", instance.url.format("127.0.0.1"), "or", instance.url.format(get_ip()), "to view in browser")
            cls.connections[name] = instance
            return instance

    def send(self, msg):
        if len(self.clients) == 0:
            display(self.iframe)
        while len(self.clients) == 0:
            time.sleep(0.1)
        for client in self.clients:
            self.__loop.add_callback(send_callback, client, msg)

    def send_t(self, method, msg):
        cmd = {"method": method, "time": time.time(), "data": msg}
        self.cache.append(cmd)
        self.send(json.dumps([cmd]))

    def clear(self):
        self.send_t("clear", "")
        self.cache = []

    def wait(self):
        input("Press enter to exit")

    @classmethod
    def run(cls):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        static_path = os.path.join(os.path.dirname(__file__), "static")
        template_path = os.path.join(os.path.dirname(__file__), ".")
        app = tornado.web.Application(
            [
                (r"/view/(.*)", IndexHandler),
                (r"/ws/(.*)", WSHandler),
                (r"/(.*)", StaticHandler),
            ],
            static_path=static_path,
            template_path=template_path,
            debug=True,
            autoreload=True,
        )
        http_server = tornado.httpserver.HTTPServer(app)
        http_server.listen(cls.port)
        cls.__loop = tornado.ioloop.IOLoop.current()
        cls.__loop.start()


class Space(Page):
    def render_point(self, index, vertice, color, size):
        self.send_t("point", {"index":index, "vertice": vertice, "color": color, "size": size})

    def render_mesh(self, index, vertice, color):
        self.send_t("mesh", {"index": index, "vertice": vertice, "color": color})

    def render_line(self, index, vertice, color):
        self.send_t("line", {"index":index,"vertice": vertice, "color": color})

    def render_text(self, index, text, x, y, z, color):
        self.send_t("text", {"index":index, "text": text, "x": x, "y": y, "z":z, "color":color})


class Log(Page):
    def info(self, *msg):
        self.send_t("info", log(*msg))

    def err(self, *msg):
        self.send_t("err", log(*msg))

    def warn(self, *msg):
        self.send_t("warn", log(*msg))

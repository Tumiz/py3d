import socket
import struct
import hashlib,base64
import webbrowser
import json

def open_viewer():
    print("open viewer")
    webbrowser.open("viewer.html",new=0,autoraise=False)

class WebSocket:
    def __init__(self,path=""):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        self.socket.bind(('127.0.0.1',52330))
        self.socket.listen(50)
        print("Listening to: 52330")
        self.conn, address = self.socket.accept()#wait for frontend's request
        self.name = address[0]
        self.remote = address
        self.path = path
        self.GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
        self.buffer = ""
        self.buffer_utf8 = b""
        self.length_buffer = 0
        self.handshaken=False
        while not self.handshaken:
            print('INFO: Socket Start Handshaken with '+str(self.remote))
            try:
                self.buffer = self.conn.recv(1024).decode('utf-8') #socket会话收到的只能是utf-8编码的信息，将接收到的bytes数据，通过utf-8编码方式解码为unicode编码进行处理
            except Exception:
                self.conn.close()
                print("socket会话收到的只能是utf-8编码的信息，将接收到的bytes数据，通过utf-8编码方式解码为unicode编码进行处理")
                print("INFO: Socket self.buffer is {%s}" % (self.buffer))
            if self.buffer.find('\r\n\r\n') != -1:
                headers = {}
                header, data = self.buffer.split('\r\n\r\n', 1) #按照这种标志分割一次,结果为：header data
                #对header进行分割后，取出后面的n-1个部分
                for line in header.split("\r\n")[1:]: #再对header 和 data部分进行单独的解析
                    key, value = line.split(": ", 1) #逐行的解析Request Header信息(Key,Value)
                    headers[key] = value
                try:
                    WebSocketKey = headers["Sec-WebSocket-Key"]
                except KeyError:
                    print("Socket Handshaken Failed!")
                    self.conn.close()
                    break
                WebSocketToken = self.generate_token(WebSocketKey)
                headers["Location"] = ("ws://%s%s" %(headers["Host"], self.path))
                #握手过程,服务器构建握手的信息,进行验证和匹配
                #Upgrade: WebSocket 表示为一个特殊的http请求,请求目的为从http协议升级到websocket协议
                handshake = "HTTP/1.1 101 Switching Protocols\r\n"\
                        "Connection: Upgrade\r\n"\
                        "Sec-WebSocket-Accept: " + WebSocketToken + "\r\n"\
                        "Upgrade: websocket\r\n\r\n"
                self.conn.send(handshake.encode(encoding='utf-8')) # 前文以bytes类型接收，此处以bytes类型进行发送
                # 此处需要增加代码判断是否成功建立连接
                self.handshaken = True #socket连接成功建立之后修改握手标志
                print("WebServer is listening "+str(self.remote) )
                #向全部连接客户端集合发送消息,(环境套接字x的到来)
                self.send_msg("Welocomg " + self.name + " !")
            else:
                print("Socket Error2!")
                self.conn.close()
                break

    # 调用socket的send方法发送str信息给web端
    def send_msg(self,msg):
        msg=json.dumps(msg)
        header = b""  #使用bytes格式,避免后面拼接的时候出现异常
        header += b"\x81"
        back_str = []
        back_str.append('\x81')
        data_length = len(msg.encode()) #可能有中文内容传入，因此计算长度的时候需要转为bytes信息
        print("INFO: send message is %s and len is %d" % (msg, len(msg.encode('utf-8'))))
        # 数据长度的三种情况
        if data_length <= 125:#当消息内容长度小于等于125时，数据帧的第二个字节0xxxxxxx 低7位直接标示消息内容的长度
            header += str.encode(chr(data_length))
        elif data_length <= 65535:#当消息内容长度需要两个字节来表示时,此字节低7位取值为126,由后两个字节标示信息内容的长度
            header += struct.pack('b', 126)
            header += struct.pack('>h', data_length)
        elif data_length <= (2^64-1):#当消息内容长度需要把个字节来表示时,此字节低7位取值为127,由后8个字节标示信息内容的长度  
            header += struct.pack('b', 127)
            header += struct.pack('>q', data_length)
        else:
            print (u'太长了')
        ass_msg = header + msg.encode('utf-8')
        self.conn.send(ass_msg)

    def generate_token(self, WebSocketKey):
        WebSocketKey = WebSocketKey + self.GUID
        Ser_WebSocketKey = hashlib.sha1(WebSocketKey.encode(encoding='utf-8')).digest()
        WebSocketToken = base64.b64encode(Ser_WebSocketKey) # 返回的是一个bytes对象
        return WebSocketToken.decode('utf-8')
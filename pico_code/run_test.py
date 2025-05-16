from wifiserver import WifiServer

server = WifiServer('batnet', "lebowski")
server.connect()
server.start()
print('start receiving')
r = server.receive()
print(r)
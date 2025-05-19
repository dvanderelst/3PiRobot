import Client

client = Client.Client('192.168.1.15')
client.set_motors(0, 0)
x = client.ping(10000, 200)
print(x)
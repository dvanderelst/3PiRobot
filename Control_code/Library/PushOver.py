import http.client, urllib
import socket

token = 'a5oxy3mohvh8bnet96khzss2mysyxy'
key = 'uzpf92mx63j8n49w4j51arzbirgi9i'

def get_computer_name():
    try:
        hostname = socket.gethostname()
        return hostname
    except Exception as e:
        return f"An error occurred: {e}"

def send(message):
    computer = get_computer_name()
    conn = http.client.HTTPSConnection("api.pushover.net:443")
    conn.request("POST", "/1/messages.json",
      urllib.parse.urlencode({
        "token": token,
        "user": key,
        "message": '[' + computer +'] ' + str(message),
      }), { "Content-type": "application/x-www-form-urlencoded" })
    result = conn.getresponse()
    return result


#r = send('test')
#print(r)
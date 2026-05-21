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
    """Send a Pushover notification. Returns the HTTPResponse with an
    extra `.body` attribute (the response body, already read and decoded),
    or None if the request raised. Non-200 responses print a warning so
    silent server-side failures don't go unnoticed."""
    computer = get_computer_name()
    try:
        conn = http.client.HTTPSConnection("api.pushover.net:443", timeout=10)
        conn.request("POST", "/1/messages.json",
          urllib.parse.urlencode({
            "token": token,
            "user": key,
            "message": '[' + computer +'] ' + str(message),
          }), { "Content-type": "application/x-www-form-urlencoded" })
        result = conn.getresponse()
        # Cache body so callers don't have to read() (single-shot) and so
        # we can include it in the warning below.
        result.body = result.read().decode("utf-8", errors="replace")
        if result.status != 200:
            print(f"PushOver.send WARNING: HTTP {result.status} -- {result.body}")
        return result
    except Exception as e:
        print(f"PushOver.send EXCEPTION: {type(e).__name__}: {e}")
        return None


#r = send('test')
#print(r)
import Client
import Process

client = Client.Client('192.168.200.38')
client.name = 'Robot01'
client.verbose = False

for i in range(3):
    print('------------------')
    data, distance_axis, timing = client.ping(10000, 150, False)
    results = Process.process(data, distance_axis, plot=True)
    distance = results['onset_distance']
    integrals = results['log_integrals']
    print('distance:', distance)
    print('integrals:', integrals)





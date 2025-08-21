from library import tools
from library import rshell
origin_folder = '/home/dieter/Dropbox/PythonRepos/3PiRobot/robot_code/upload'

ports = tools.scan_for_ports()
ports = tools.select_robot_ports(ports)

for port in ports:
    rshell.make_staging_copy(origin_folder)
    rshell.remove_same(port, 'staging')
    rshell.upload(port, 'staging')

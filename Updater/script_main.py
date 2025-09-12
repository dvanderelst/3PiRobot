from library import tools
from library import rshell
origin_folder = '/home/dieter/Dropbox/PythonRepos/3PiRobot/Robot_code/upload'
staging_folder = 'staging'
ports = tools.scan_for_ports()
ports = tools.select_robot_ports(ports)

full_update = False

rshell.make_staging_copy(origin_folder, full=full_update)
for port in ports:
    if full_update: rshell.remove_same(port, staging_folder)
    rshell.upload(port, 'staging', mirror=full_update)

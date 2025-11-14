from PyLorex.library import ServerClient
from Library import Logging
from Library import Settings
from Library import Utils

# This is a wrapper around the TelemetryClient to provide logging
# and easy access to the telemetry server.

class LorexTracker:
    def __init__(self):
        self.server_client = ServerClient.TelemetryClient()

    def print_message(self, message, category):
        Logging.print_message('Tracker', message, category)

    def get_position(self, robot_number, simple=True):
        index = robot_number - 1
        configuration = Settings.get_client_config(index)
        aruco_id = configuration.aruco_id
        robot_name = configuration.robot_name

         # Get the position from the server
        camera_name, x, y, yaw = self.server_client.get_tracker(aruco_id)
        if x is None:
            message = f"ID {aruco_id}, {robot_name}> C:{camera_name}, Not Found"
            message_type = "WARNING"
        else:
            message = f"ID {aruco_id}, {robot_name}> C:{camera_name}, X:{int(x)}, Y:{int(y)}, Yaw:{int(yaw)}"
            message_type = "INFO"

        yaw = Utils.wrap_angle(yaw)

        self.print_message(message, message_type)
        if simple: return x, y, yaw
        package = {'camera_name': camera_name, 'x': x, 'y': y, 'yaw': yaw}
        return package

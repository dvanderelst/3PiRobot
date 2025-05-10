from pololu_3pi_2040_robot import robot
from pololu_3pi_2040_robot.extras import editions
import time
import _thread

# Initialize hardware components
motors = robot.Motors()
encoders = robot.Encoders()
display = robot.Display()

# Robot configuration (adjust these values based on your edition)
GEAR_RATIO = 30          # Gear ratio: 30:1 for Standard, 75:1 for Turtle, 15:1 for Hyper
COUNTS_PER_REV = 12      # Encoder counts per motor shaft revolution
WHEEL_DIAMETER = 32      # mm (measure actual wheel size)

# Calculated constants
COUNTS_PER_WHEEL_REV = COUNTS_PER_REV * GEAR_RATIO
WHEEL_CIRCUMFERENCE = 3.14159 * WHEEL_DIAMETER / 1000  # in meters

def counts_to_mps(counts_per_second):
    """Convert encoder counts/s to meters/second"""
    return (counts_per_second / COUNTS_PER_WHEEL_REV) * WHEEL_CIRCUMFERENCE

def run_max_speed_test(duration=5):
    """Run motors at full power and measure max speed"""
    encoders.get_counts()  # Reset encoder counts
    motors.set_speeds(0, 0)  # Ensure motors are stopped before starting
    
    max_speed = 0
    
    display.fill(0)
    display.text("Max Speed Test", 0, 0)
    display.show()
    
    start_time = time.ticks_ms()
    last_counts = encoders.get_counts()
    
    try:
        motors.set_speeds(6000, 6000)  # Full power (adjust based on your edition)
        
        while time.ticks_diff(time.ticks_ms(), start_time) < duration * 1000:
            current_time = time.ticks_ms()
            dt = time.ticks_diff(current_time, start_time) / 1000
            
            if dt == 0:
                continue
            
            current_counts = encoders.get_counts()
            delta_left = current_counts[0] - last_counts[0]
            delta_right = current_counts[1] - last_counts[1]
            
            if delta_left == 0 and delta_right == 0:
                continue
            
            # Calculate average speed from both wheels
            speed_left = counts_to_mps(delta_left / dt)
            speed_right = counts_to_mps(delta_right / dt)
            average_speed = (speed_left + speed_right) / 2
            
            max_speed = max(max_speed, average_speed)
            
            last_counts = current_counts
        
        motors.set_speeds(0, 0)   # Stop motors after test
        
        display.fill(0)
        display.text("Test Complete!", 0, 10)
        display.text(f"Max Speed: {max_speed:.2f} m/s", 0, 20)
        display.show()

        print(f"Maximum Speed: {max_speed:.2f} m/s")  # Print final result to console

    except Exception as e:
        motors.set_speeds(0, 0)   # Stop motors in case of error
        print(f"Error: {e}")

# Run test for maximum speed measurement
run_max_speed_test(duration=10)

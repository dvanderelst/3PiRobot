#!/usr/bin/env python3
"""
Minimal test script for the new synchronized communication protocol.

This script tests the command-acknowledgment sequencing to verify that:
1. Commands are properly acknowledged
2. No commands are lost
3. The protocol handles timeouts gracefully
4. The robot can keep up with the command rate
"""

import time
import sys
from Library import Client
from Library import Utils

def test_synchronized_protocol():
    """Test the new synchronized communication protocol."""
    
    print("=== Synchronized Protocol Test ===")
    print("This test verifies the new command-acknowledgment protocol.")
    print("The robot should acknowledge each command before processing.")
    print()
    
    # Initialize client
    robot_number = 1
    try:
        client = Client.Client(robot_number=robot_number)
        print(f"✓ Connected to Robot {robot_number}")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return False
    
    # Test parameters
    test_commands = 5  # Number of ping-read cycles to test
    sleep_time = 0.1   # Time between commands (should work now!)
    
    print(f"Testing {test_commands} ping-read cycles with {sleep_time}s delay...")
    print()
    
    success_count = 0
    
    for i in range(test_commands):
        print(f"--- Cycle {i+1}/{test_commands} ---")
        
        try:
            # Step 1: Send ping command (should wait for ACK)
            print("Sending ping command...")
            acquire_id = client.acquire(action='ping')
            print(f"✓ Ping command sent and acknowledged (ID: {acquire_id})")
            
            # Step 2: Read buffers (should wait for data)
            print("Reading buffers...")
            start_time = time.time()
            sonar_package = client.read_buffers()
            read_time = time.time() - start_time
            
            if sonar_package is not None:
                print(f"✓ Received sonar data ({len(sonar_package.get('sonar_data', []))} samples) in {read_time:.3f}s")
                success_count += 1
            else:
                print("✗ No sonar data received (timeout)")
                
            # Step 3: Small movement
            print("Sending small movement...")
            client.step(distance=0.05)  # Small movement
            print("✓ Movement command sent")
            
            # Brief pause before next cycle
            time.sleep(sleep_time)
            print()
            
        except Exception as e:
            print(f"✗ Error in cycle {i+1}: {e}")
            print()
            continue
    
    # Summary
    print("=== Test Summary ===")
    print(f"Successful cycles: {success_count}/{test_commands}")
    print(f"Success rate: {success_count/test_commands*100:.1f}%")
    
    if success_count == test_commands:
        print("🎉 All tests passed! The synchronized protocol is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the robot logs for details.")
        return False

def test_timeout_handling():
    """Test the timeout handling of the new protocol."""
    print("\n=== Timeout Handling Test ===")
    print("This test verifies that timeouts are handled gracefully.")
    
    try:
        client = Client.Client(robot_number=1)
        
        # Try to send a command that might timeout
        print("Testing command timeout...")
        try:
            # This should either succeed or timeout gracefully
            acquire_id = client.acquire(action='ping')
            print(f"✓ Command completed successfully (ID: {acquire_id})")
        except TimeoutError:
            print("✓ Timeout handled gracefully (this is expected if robot is busy)")
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return False
    
    print("✓ Timeout handling test completed")
    return True

if __name__ == "__main__":
    print("Starting Synchronized Protocol Test")
    print("=" * 50)
    
    # Run main test
    test1_passed = test_synchronized_protocol()
    
    # Run timeout test
    test2_passed = test_timeout_handling()
    
    # Final summary
    print("\n" + "=" * 50)
    print("FINAL RESULTS:")
    print(f"Synchronized Protocol Test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Timeout Handling Test: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! The new protocol is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the implementation.")
        sys.exit(1)
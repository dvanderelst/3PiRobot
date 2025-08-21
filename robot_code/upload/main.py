import run_tests

# Specify the tests to run
# 0 = WiFi Test
# 1 = LED Test
# 2 = Screen Test
# 3 = Bumper Test
# 4 = Sonar Test
# 5 = Drive Test

tests = [1]
run_tests.run_tests(tests)
if len(tests) == 0: import run_default
import run_tests
do_run_tests = False

while do_run_tests:
    selected_tests = run_tests.select_tests()
    run_tests.run_tests(selected_tests)

import run_default
import Simulate
import Equation
from matplotlib import pyplot as plt

diameter_3d = 5
height_3d = 1
n_points_3d = 25

diameter_2d = 2.5
n_points_2d_list = list(range(1, 11))

result_3d = Simulate.run_3d(n_points_3d, diameter_3d, height_3d)
result_2d = []
for n in n_points_2d_list:
    result = Simulate.run_2d(n, diameter_2d)
    result_2d.append(result)

N2 = Equation.bats_3d_to_2d(diameter_3d, height_3d, n_points_3d, diameter_2d)



plt.figure()
plt.plot(n_points_2d_list, result_2d, marker='o', label='2D mean nearest distance')
plt.axhline(result_3d, color='red', linestyle='--', label='3D mean nearest distance')
plt.axvline(N2, color='green', linestyle='--', label=f'Equivalent 2D count: {N2:.2f}')
plt.xlabel('Number of points in 2D')
plt.ylabel('Mean nearest distance')
plt.title('Mean Nearest Distance in 2D vs 3D')
plt.legend()
plt.grid()
plt.show()


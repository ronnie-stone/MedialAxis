import numpy as np
from ortools.linear_solver import pywraplp
from generate_task_list import generate_task_lists
import matplotlib.pyplot as plt


def task_scheduling_ilp(tasks, task_durations, robots, connected_tasks, verbose=False, num_time_slots=1000):
    # Create the solver instance
    solver = pywraplp.Solver.CreateSolver('CBC')
    #solver.SetSolverSpecificParametersAsString("numerics/feastol = 1e-5")

    if not solver:
        return None
    
    try:
    
        num_tasks = len(tasks)
        M = num_time_slots  # Large constant to handle non-overlap constraints

        # Decision variables
        s = {}
        z = {}

        for i in range(num_tasks):
            # Allow start times to be any positive real number
            s[i] = solver.NumVar(0, solver.infinity(), f'start_time_{i}')
            for j in range(i + 1, num_tasks):
                # z is still binary as it controls task sequencing
                z[i, j] = solver.BoolVar(f'z_{i}_{j}')
        
        # Variable for makespan, also a continuous variable
        T = solver.NumVar(0, solver.infinity(), 'makespan')

        # Constraints: No overlapping tasks on the same robot
        for robot in set(robots):
            tasks_on_robot = [i for i in range(num_tasks) if robots[i] == robot]
            for i in range(len(tasks_on_robot)):
                for j in range(i + 1, len(tasks_on_robot)):
                    task_i, task_j = tasks_on_robot[i], tasks_on_robot[j]
                    solver.Add(s[task_i] + task_durations[task_i] <= s[task_j] + M * (1 - z[task_i, task_j]))
                    solver.Add(s[task_j] + task_durations[task_j] <= s[task_i] + M * z[task_i, task_j])

        # Constraints: No overlapping connected tasks
        for i, j in connected_tasks:
            solver.Add(s[i] + task_durations[i] <= s[j] + M * (1 - z[i, j]))
            solver.Add(s[j] + task_durations[j] <= s[i] + M * z[i, j])

        # Constraints: All tasks must complete by the makespan T
        for i in range(num_tasks):
            solver.Add(s[i] + task_durations[i] <= T)

        # Objective: Minimize the makespan
        solver.Minimize(T)

        # Solve the model
        status = solver.Solve()
        if status != pywraplp.Solver.OPTIMAL:
            print("No optimal solution found")

        if verbose:
            if status == pywraplp.Solver.OPTIMAL:
                print(f'Optimal makespan: {T.solution_value():.3f}')
                print(f'Detailed Schedule:')
                for i in range(num_tasks):
                    start_time = s[i].solution_value()
                    end_time = start_time + task_durations[i]
                    print(f'Task {i} assigned to Robot {robots[i]} starts at {start_time:.3f}, lasts for {task_durations[i]:.3f} units, and ends at {end_time:.3f}.')
            else:
                print('No optimal solution found.')
            print("")

        return T.solution_value()
    
    except:

        return 100

if __name__ == "__main__":

    # Example input (4 printers)

    tasks = [0, 1, 2, 3, 4, 5, 6, 7]
    task_durations = [1, 1, 1, 1, 5, 5, 5, 5]  # Time required for each task
    robots = [1, 2, 3, 4, 1, 2, 3, 4]  # Task assignments to robots
    connected_tasks_1 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # Tasks that share physical boundaries and cannot be done simultaneously
    connected_tasks_2 = [(0, 1), (0, 3), (1, 2), (1, 3), (2, 3)]  # Tasks that share physical boundaries and cannot be done simultaneously
    connected_tasks_3 = [(0, 1), (0, 3), (1, 2), (2, 3)]  # Tasks that share physical boundaries and cannot be done simultaneously

    makespan_1 = task_scheduling_ilp(tasks, task_durations, robots, connected_tasks_2, verbose=True)
    makespan_2 = task_scheduling_ilp(tasks, task_durations, robots, connected_tasks_3)
    print(makespan_2-makespan_1)

    # Test graph coloring hypothesis.

    # k = 100
    # T = 1
    # n = len(tasks)
    # task_lists = generate_task_lists(T, n, k)

    # lists_of_numbers = []
    # for i in range(k):
    #     task_durations = task_lists[i]
    #     makespan_4_colorable = task_scheduling_ilp(tasks, task_durations, robots, connected_tasks_1)
    #     makespan_2_colorable = task_scheduling_ilp(tasks, task_durations, robots, connected_tasks_3)
    #     print(i, makespan_2_colorable - makespan_4_colorable)
    #     if makespan_2_colorable - makespan_4_colorable > 0.0001:
    #         print(i, "False")
    #         print(makespan_2_colorable, makespan_4_colorable)
    #         print(task_durations)
    #         break
    #     if abs(makespan_2_colorable - makespan_4_colorable) < 0.0001:
    #         lists_of_numbers.append(task_durations)
    #         break

    # Example input (5 printers)

    # tasks = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # # task_durations = np.array([1, 1, 1, 1, 1, 5, 5, 5, 5, 5], dtype="float64")  # Time required for each task
    # robots = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]  # Task assignments to robots

    # connected_tasks_1 = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]  # Tasks that share physical boundaries and cannot be done simultaneously
    # connected_tasks_2 = [(0, 1), (0, 3), (0, 4), (1, 2), (1, 3), (2, 3), (3, 4)]  # Tasks that share physical boundaries and cannot be done simultaneously
    # #connected_tasks_3 = [(0, 1), (0, 3), (0, 4), (1, 2), (1, 3), (2, 3), (3, 4)]

    # # Test graph coloring hypothesis.

    # k = 1000
    # T = 10
    # n = len(tasks)
    # task_lists = generate_task_lists(T, n, k)

    # for i in range(k):
    #     task_durations = task_lists[i]
    #     makespan_4_colorable = task_scheduling_ilp(tasks, task_durations, robots, connected_tasks_1)
    #     makespan_3_colorable = task_scheduling_ilp(tasks, task_durations, robots, connected_tasks_2)
    #     print(i, makespan_3_colorable - makespan_4_colorable)
    #     if makespan_3_colorable - makespan_4_colorable > 0.0001:
    #         print(i, " False")
    #         print(makespan_3_colorable, makespan_4_colorable)
    #         print(task_durations)
    #         break

    # Run the ILP scheduling
    # task_scheduling_ilp(tasks, task_durations, robots, connected_tasks_1)
    # task_scheduling_ilp(tasks, task_durations, robots, connected_tasks_2)

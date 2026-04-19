import random
import math
import heapq
import statistics


# --------------------------------------------------------------------------
# Maze Setup
def coord_to_node(x, y):
    return x * 6 + y


def node_to_coord(node):
    return (node // 6, node % 6)


def generate_random_maze():
    all_nodes = list(range(36))

    start = random.choice(range(0, 12))
    goal = random.choice(range(24, 36))

    remaining = [n for n in all_nodes if n not in (start, goal)]
    barriers = set(random.sample(remaining, 4))

    return {
        "start": start,
        "goal": goal,
        "barriers": barriers
    }


def print_maze(maze):
    print("\n6x6 Maze:\n")
    for y in range(6):
        row = []
        for x in range(6):
            node = coord_to_node(x, y)

            if node == maze["start"]:
                row.append(" S ")
            elif node == maze["goal"]:
                row.append(" G ")
            elif node in maze["barriers"]:
                row.append(" X ")
            else:
                row.append(f"{node:2d} ")
        print(" ".join(row))


# -------------------------------------------------------------------------
# Neighbors, Moves, Edge Cost
directions = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1)
]


def is_valid_coord(x, y):
    return 0 <= x < 6 and 0 <= y < 6


def get_neighbors(node, barriers):
    x, y = node_to_coord(node)
    neighbors = []

    for dx, dy in directions:
        nx, ny = x + dx, y + dy

        if is_valid_coord(nx, ny):
            next_node = coord_to_node(nx, ny)

            if next_node not in barriers:
                neighbors.append(next_node)

    return sorted(neighbors)


def edge_cost(node1, node2):
    x1, y1 = node_to_coord(node1)
    x2, y2 = node_to_coord(node2)

    dx = abs(x1 - x2)
    dy = abs(y1 - y2)

    if dx == 1 and dy == 1:
        return math.sqrt(2)
    elif dx + dy == 1:
        return 1
    else:
        return math.inf


def calculate_path_cost(path):
    if not path or len(path) < 2:
        return 0

    total = 0
    for i in range(len(path) - 1):
        total += edge_cost(path[i], path[i + 1])
    return total


# ------------------------------------------------------------------------
# Depth-Limited DFS
def depth_limited_dfs(current, goal, barriers, limit, path, visited_order):
    visited_order.append(current)

    if current == goal:
        return path

    if limit == 0:
        return None

    for neighbor in get_neighbors(current, barriers):
        if neighbor not in path:  # avoid cycles
            result = depth_limited_dfs(
                neighbor,
                goal,
                barriers,
                limit - 1,
                path + [neighbor],
                visited_order
            )
            if result is not None:
                return result

    return None


# -------------------------------------------------------------------------
# Iterative Deepening DFS
def iterative_deepening_dfs(start, goal, barriers, max_depth=20):
    total_visited_order = []

    for depth in range(max_depth + 1):
        visited_order = []

        result = depth_limited_dfs(
            start,
            goal,
            barriers,
            limit=depth,
            path=[start],
            visited_order=visited_order
        )

        total_visited_order.extend(visited_order)

        if result is not None:
            return {
                "path": result,
                "visited": total_visited_order,
                "time": len(total_visited_order)
            }

    return {
        "path": None,
        "visited": total_visited_order,
        "time": len(total_visited_order)
    }


# -------------------------------------------------------------------------
# Heuristics
def chebyshev_distance(node, goal):
    x1, y1 = node_to_coord(node)
    x2, y2 = node_to_coord(goal)
    return max(abs(x1 - x2), abs(y1 - y2))


def manhattan_distance(node, goal):
    x1, y1 = node_to_coord(node)
    x2, y2 = node_to_coord(goal)
    return abs(x1 - x2) + abs(y1 - y2)


def euclidean_distance(node, goal):
    x1, y1 = node_to_coord(node)
    x2, y2 = node_to_coord(goal)
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def diagonal_distance(node, goal):
    x1, y1 = node_to_coord(node)
    x2, y2 = node_to_coord(goal)

    dx = abs(x1 - x2)
    dy = abs(y1 - y2)

    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)


# ------------------------------------------------------------------------
# Best First Search
def best_first_search(start, goal, barriers, heuristic_fn):
    frontier = []
    heapq.heappush(frontier, (heuristic_fn(start, goal), start, [start]))

    visited_order = []
    explored = set()

    while frontier:
        h_value, current, path = heapq.heappop(frontier)

        if current in explored:
            continue

        explored.add(current)
        visited_order.append(current)

        if current == goal:
            return {
                "path": path,
                "visited": visited_order,
                "time": len(visited_order)
            }

        for neighbor in get_neighbors(current, barriers):
            if neighbor not in explored:
                new_path = path + [neighbor]
                heapq.heappush(
                    frontier,
                    (heuristic_fn(neighbor, goal), neighbor, new_path)
                )

    return {
        "path": None,
        "visited": visited_order,
        "time": len(visited_order)
    }


# ------------------------------------------------------------------------
# Output Helpers
def print_search_result(name, result):
    print(f"\n--- {name} ---")
    print("Visited nodes:", result["visited"])
    print("Number of visited nodes:", len(result["visited"]))
    print("Time to find goal:", result["time"], "minutes")
    print("Final path:", result["path"])
    print("Path length:", len(result["path"]) if result["path"] else 0)
    print("Path cost:", calculate_path_cost(result["path"]))


def print_trial_result(trial_number, trial_data):
    maze = trial_data["maze"]

    print(f"\n{'-' * 20} Trial {trial_number} {'-' * 20}")
    print("Start node:", maze["start"], "Coordinates:", node_to_coord(maze["start"]))
    print("Goal node:", maze["goal"], "Coordinates:", node_to_coord(maze["goal"]))
    print("Barrier nodes:", sorted(maze["barriers"]))

    print_maze(maze)

    print_search_result("IDDFS", trial_data["iddfs"])
    print_search_result("Best First (Chebyshev)", trial_data["chebyshev"])
    print_search_result("Best First (Manhattan)", trial_data["manhattan"])
    print_search_result("Best First (Euclidean)", trial_data["euclidean"])
    print_search_result("Best First (Diagonal)", trial_data["diagonal"])


# ------------------------------------------------------------------------
# Trial Runner
def run_single_trial():
    maze = generate_random_maze()

    iddfs_result = iterative_deepening_dfs(
        maze["start"],
        maze["goal"],
        maze["barriers"]
    )

    chebyshev_result = best_first_search(
        maze["start"], maze["goal"], maze["barriers"], chebyshev_distance
    )

    manhattan_result = best_first_search(
        maze["start"], maze["goal"], maze["barriers"], manhattan_distance
    )

    euclidean_result = best_first_search(
        maze["start"], maze["goal"], maze["barriers"], euclidean_distance
    )

    diagonal_result = best_first_search(
        maze["start"], maze["goal"], maze["barriers"], diagonal_distance
    )

    return {
        "maze": maze,
        "iddfs": iddfs_result,
        "chebyshev": chebyshev_result,
        "manhattan": manhattan_result,
        "euclidean": euclidean_result,
        "diagonal": diagonal_result
    }


# ------------------------------------------------------------------------
# Summary Helpers
def extract_summary_data(trials, algo_key):
    times = []
    path_lengths = []
    path_costs = []

    for trial in trials:
        result = trial[algo_key]
        if result["path"] is not None:
            times.append(result["time"])
            path_lengths.append(len(result["path"]))
            path_costs.append(calculate_path_cost(result["path"]))

    return times, path_lengths, path_costs


def calculate_statistics(values):
    if not values:
        return None, None

    mean_value = statistics.mean(values)
    variance_value = statistics.variance(values) if len(values) > 1 else 0
    return mean_value, variance_value


def print_stats(name, times, lengths, costs):
    mean_time, var_time = calculate_statistics(times)
    mean_len, var_len = calculate_statistics(lengths)
    mean_cost, var_cost = calculate_statistics(costs)

    print(f"\n{name}")
    print("Times:", times)
    print("Mean time:", mean_time)
    print("Variance of time:", var_time)

    print("Path lengths:", lengths)
    print("Mean path length:", mean_len)
    print("Variance of path length:", var_len)

    print("Path costs:", costs)
    print("Mean path cost:", mean_cost)
    print("Variance of path cost:", var_cost)


def print_comparison_table(all_stats):
    print("\n" + "-" * 70)
    print("COMPARISON TABLE")
    print("-" * 70)
    print(f"{'Algorithm':<22} {'Mean Time':<12} {'Var Time':<12} {'Mean Len':<12} {'Mean Cost':<12}")
    print("-" * 70)

    for name, stats_dict in all_stats.items():
        print(
            f"{name:<22} "
            f"{stats_dict['mean_time']:<12.4f} "
            f"{stats_dict['var_time']:<12.4f} "
            f"{stats_dict['mean_len']:<12.4f} "
            f"{stats_dict['mean_cost']:<12.4f}"
        )


def find_best_algorithms(all_stats):
    min_time = min(v["mean_time"] for v in all_stats.values())
    winners = [k for k, v in all_stats.items() if v["mean_time"] == min_time]

    print("\nFastest algorithm(s):", ", ".join(winners))
    print("Average time:", min_time)

    min_len = min(v["mean_len"] for v in all_stats.values())
    best_len = [k for k, v in all_stats.items() if v["mean_len"] == min_len]

    print("\nShortest average path length algorithm(s):", ", ".join(best_len))
    print("Average path length:", min_len)

    min_cost = min(v["mean_cost"] for v in all_stats.values())
    best_cost = [k for k, v in all_stats.items() if v["mean_cost"] == min_cost]

    print("\nLowest average path cost algorithm(s):", ", ".join(best_cost))
    print("Average path cost:", min_cost)


# ------------------------------------------------------------------------
# MAIN RUN
trials = []

for i in range(3):
    trial = run_single_trial()
    trials.append(trial)
    print_trial_result(i + 1, trial)

# Collect all summary data
iddfs_times, iddfs_lengths, iddfs_costs = extract_summary_data(trials, "iddfs")
cheb_times, cheb_lengths, cheb_costs = extract_summary_data(trials, "chebyshev")
man_times, man_lengths, man_costs = extract_summary_data(trials, "manhattan")
euc_times, euc_lengths, euc_costs = extract_summary_data(trials, "euclidean")
diag_times, diag_lengths, diag_costs = extract_summary_data(trials, "diagonal")

print("\n" + "-" * 30)
print("SUMMARY ACROSS 3 TRIALS")
print("-" * 30)

print_stats("IDDFS", iddfs_times, iddfs_lengths, iddfs_costs)
print_stats("Chebyshev", cheb_times, cheb_lengths, cheb_costs)
print_stats("Manhattan", man_times, man_lengths, man_costs)
print_stats("Euclidean", euc_times, euc_lengths, euc_costs)
print_stats("Diagonal", diag_times, diag_lengths, diag_costs)

# Build detailed stats dictionary
all_stats = {
    "IDDFS": {
        "mean_time": calculate_statistics(iddfs_times)[0],
        "var_time": calculate_statistics(iddfs_times)[1],
        "mean_len": calculate_statistics(iddfs_lengths)[0],
        "var_len": calculate_statistics(iddfs_lengths)[1],
        "mean_cost": calculate_statistics(iddfs_costs)[0],
        "var_cost": calculate_statistics(iddfs_costs)[1],
    },
    "Chebyshev": {
        "mean_time": calculate_statistics(cheb_times)[0],
        "var_time": calculate_statistics(cheb_times)[1],
        "mean_len": calculate_statistics(cheb_lengths)[0],
        "var_len": calculate_statistics(cheb_lengths)[1],
        "mean_cost": calculate_statistics(cheb_costs)[0],
        "var_cost": calculate_statistics(cheb_costs)[1],
    },
    "Manhattan": {
        "mean_time": calculate_statistics(man_times)[0],
        "var_time": calculate_statistics(man_times)[1],
        "mean_len": calculate_statistics(man_lengths)[0],
        "var_len": calculate_statistics(man_lengths)[1],
        "mean_cost": calculate_statistics(man_costs)[0],
        "var_cost": calculate_statistics(man_costs)[1],
    },
    "Euclidean": {
        "mean_time": calculate_statistics(euc_times)[0],
        "var_time": calculate_statistics(euc_times)[1],
        "mean_len": calculate_statistics(euc_lengths)[0],
        "var_len": calculate_statistics(euc_lengths)[1],
        "mean_cost": calculate_statistics(euc_costs)[0],
        "var_cost": calculate_statistics(euc_costs)[1],
    },
    "Diagonal": {
        "mean_time": calculate_statistics(diag_times)[0],
        "var_time": calculate_statistics(diag_times)[1],
        "mean_len": calculate_statistics(diag_lengths)[0],
        "var_len": calculate_statistics(diag_lengths)[1],
        "mean_cost": calculate_statistics(diag_costs)[0],
        "var_cost": calculate_statistics(diag_costs)[1],
    }
}

print_comparison_table(all_stats)
find_best_algorithms(all_stats)

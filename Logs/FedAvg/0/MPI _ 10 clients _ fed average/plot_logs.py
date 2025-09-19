import re
import matplotlib.pyplot as plt

# Path to your log file
log_file = "log_rank0.txt"

# Containers
rounds, losses, accs = [], [], []
wall_times = []

# Regex patterns
round_pattern = re.compile(r"Round\s+(\d+)\s+\|\s+Test Loss:\s+([\d\.]+)\s+\|\s+Test Acc:\s+([\d\.]+)%")
walltime_pattern = re.compile(r"Total wall time per rank \(s\): \[([^\]]+)\]")

with open(log_file, "r") as f:
    for line in f:
        # Match rounds
        match = round_pattern.search(line)
        if match:
            rounds.append(int(match.group(1)))
            losses.append(float(match.group(2)))
            accs.append(float(match.group(3)))
        
        # Match wall times
        match2 = walltime_pattern.search(line)
        if match2:
            wall_times = [float(x.strip()) for x in match2.group(1).split(",")]

# ---- Plotting ----
plt.figure(figsize=(10,6))
plt.plot(rounds, losses, label="Test Loss", color="red")
plt.xlabel("Rounds")
plt.ylabel("Loss")
plt.title("Test Loss vs Rounds")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,6))
plt.plot(rounds, accs, label="Test Accuracy", color="blue")
plt.xlabel("Rounds")
plt.ylabel("Accuracy (%)")
plt.title("Test Accuracy vs Rounds")
plt.legend()
plt.grid(True)
plt.show()

# Combined plot (dual axis)
fig, ax1 = plt.subplots(figsize=(10,6))
ax2 = ax1.twinx()
ax1.plot(rounds, losses, "r-", label="Loss")
ax2.plot(rounds, accs, "b-", label="Accuracy")

ax1.set_xlabel("Rounds")
ax1.set_ylabel("Loss", color="r")
ax2.set_ylabel("Accuracy (%)", color="b")
plt.title("Loss & Accuracy vs Rounds")
fig.tight_layout()
plt.show()

# Wall times bar chart
if wall_times:
    plt.figure(figsize=(10,6))
    plt.bar(range(len(wall_times)), wall_times, color="green")
    plt.xlabel("Rank")
    plt.ylabel("Wall Time (s)")
    plt.title("Wall Time per Rank")
    plt.grid(axis="y")
    plt.show()

    # Histogram of wall times
    plt.figure(figsize=(10,6))
    plt.hist(wall_times, bins=10, color="purple", alpha=0.7)
    plt.xlabel("Wall Time (s)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Wall Times")
    plt.grid(True)
    plt.show()

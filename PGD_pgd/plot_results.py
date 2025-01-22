import re
import matplotlib.pyplot as plt

def parse_log(file_path):
    """Parse the log file and extract relevant metrics."""
    iterations = []
    l_rel = []
    l_dis = []
    avg_max_prob = []
    avg_top_p_99 = []

    with open(file_path, 'r') as file:
        for line in file:
            # Match the main metrics line
            match = re.match(r"\[(\d+)\] L-rel: ([\d.]+) / L-dis: ([\d.]+)", line)
            if match:
                iterations.append(int(match.group(1)))
                l_rel.append(float(match.group(2)))
                l_dis.append(float(match.group(3)))
            # Match the Avg Max Prob line
            if "Avg Max Prob" in line:
                avg_max_prob_match = re.search(r"Avg Max Prob: ([\d.]+)", line)
                if avg_max_prob_match:
                    avg_max_prob.append(float(avg_max_prob_match.group(1)))
            # Match the Avg Top P-99 line
            if "Avg Top P-99" in line:
                avg_top_p_99_match = re.search(r"Avg Top P-99: ([\d.]+)", line)
                if avg_top_p_99_match:
                    avg_top_p_99.append(float(avg_top_p_99_match.group(1)))

    return iterations, l_rel, l_dis, avg_max_prob, avg_top_p_99

def plot_metrics(local_metrics, remote_metrics):
    """Plot the metrics for local and remote models."""
    iterations, local_l_rel, local_l_dis, local_avg_max_prob, local_avg_top_p_99 = local_metrics
    _, remote_l_rel, remote_l_dis, remote_avg_max_prob, remote_avg_top_p_99 = remote_metrics

    plt.figure(figsize=(12, 8))

    # Plot L-rel
    plt.subplot(2, 2, 1)
    plt.plot(iterations, local_l_rel, label='Local L-rel', color='blue')
    plt.plot(iterations, remote_l_rel, label='Remote L-rel', color='orange')
    plt.xlabel('Iterations')
    plt.ylabel('L-rel')
    plt.title('L-rel Comparison')
    plt.legend()

    # Plot L-dis
    plt.subplot(2, 2, 2)
    plt.plot(iterations, local_l_dis, label='Local L-dis', color='blue')
    plt.plot(iterations, remote_l_dis, label='Remote L-dis', color='orange')
    plt.xlabel('Iterations')
    plt.ylabel('L-dis')
    plt.title('L-dis Comparison')
    plt.legend()

    # Plot Avg Max Prob
    plt.subplot(2, 2, 3)
    plt.plot(iterations, local_avg_max_prob, label='Local Avg Max Prob', color='blue')
    plt.plot(iterations, remote_avg_max_prob, label='Remote Avg Max Prob', color='orange')
    plt.xlabel('Iterations')
    plt.ylabel('Avg Max Prob')
    plt.title('Avg Max Prob Comparison')
    plt.legend()

    # Plot Avg Top P-99
    plt.subplot(2, 2, 4)
    plt.plot(iterations, local_avg_top_p_99, label='Local Avg Top P-99', color='blue')
    plt.plot(iterations, remote_avg_top_p_99, label='Remote Avg Top P-99', color='orange')
    plt.xlabel('Iterations')
    plt.ylabel('Avg Top P-99')
    plt.title('Avg Top P-99 Comparison')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Paths to the log files
local_log_file = "local_llama_attack.txt"
remote_log_file = "remote_llama_attack.txt"

# Parse the logs
local_metrics = parse_log(local_log_file)
remote_metrics = parse_log(remote_log_file)

# Plot the metrics
plot_metrics(local_metrics, remote_metrics)

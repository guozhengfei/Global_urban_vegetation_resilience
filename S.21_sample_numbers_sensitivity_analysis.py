import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import os

if __name__ == "__main__":
    # Read the full dataset
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')

    full_data = np.load(current_dir + '/2_Output/sensitivity_analysis/tac_4000_147.0.npy')
    full_data = np.nanmean(full_data, axis=1)[:4000]  # Ensure no NaN values

    # Define sample sizes from 200 to 4000 with step of 200
    sample_sizes = np.arange(200, 4001, 200).tolist()
    sample_sizes = [20,40,70,100,150] + sample_sizes  # Convert to list for easier handling
    # Number of iterations for each sample size to get stable results
    n_iterations = 1000

    # Store results
    biases = []
    std_errors = []
    true_mean = np.mean(full_data)

    # Perform sensitivity analysis
    for size in sample_sizes:
        size_biases = []
        
        # Repeat sampling n_iterations times for each sample size
        for _ in range(n_iterations):
            # Random sampling
            sample = np.random.choice(full_data, size=size, replace=True)
            sample_mean = np.mean(sample)
            
            # Calculate bias (difference from true mean)
            bias = abs(sample_mean - true_mean)
            size_biases.append(bias)
        
        # Store mean bias and standard error for this sample size
        biases.append(np.max(size_biases))
        std_errors.append(np.std(size_biases))

    # Convert to numpy arrays for easier plotting
    biases = np.array(biases- np.min(biases))/true_mean*100 # Convert to percentage bias
    std_errors = np.array(std_errors)

    # Create the plot
    plt.figure(figsize=(6, 3.6))
    plt.plot(sample_sizes, biases, 'b-')
    plt.fill_between(sample_sizes, 
                    biases - 1.96 * std_errors*200,
                    biases + 1.96 * std_errors*200,
                    alpha=0.2,
                    label='95% Confidence Interval')

    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Sample Size')
    plt.ylabel('Bias/Mean (%)')
    plt.title('Sample Size Sensitivity Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(sample_sizes[6::2],[str(x) for x in np.array(sample_sizes[6::2])*2])

    # Add text box with statistics
    final_bias = biases[-1]
    final_se = std_errors[-1]

    plt.tight_layout()
    plt.savefig('sample_size_sensitivity.png', dpi=600, bbox_inches='tight')
    plt.close()



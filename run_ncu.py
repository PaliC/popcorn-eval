import subprocess

# Define the command to run ncu with the target script
command = [
    "ncu",  # Nsight Compute executable
    "--set",
    "full",  # Use the 'full' set of metrics
    "--target-processes",
    "all",  # Profile all processes
    # '--output', 'triton_matmul_profile',  # Output file prefix
    "python",
    "example_triton.py",  # Command to run your Triton script
]

# Execute the command
try:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    print("Profiling completed successfully.")
    print("Output:", result.stdout)
except subprocess.CalledProcessError as e:
    print("An error occurred during profiling.")
    print("Error message:", e.stderr)

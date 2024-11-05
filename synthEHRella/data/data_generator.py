import subprocess
import os
import sys

class DataGenerator:
    def __init__(self, method):
        """Initialize the DataGenerator with the path to the method folder."""
        self.method = method

    def generate(self, params):
        """Run the external method to generate synthetic data."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        generate_script = os.path.join(f"{base_dir}/scripts", f'gen_{self.method}.sh')  # Assuming it's a shell script
        if not os.path.exists(generate_script):
            raise FileNotFoundError(f"Generation script not found in {self.method_folder}")
        
        # Prepare command to execute the script with parameters if needed
        command = ["bash", generate_script] + self._build_command_line_args(params)
        print(" ".join(command))
        try:
            # Execute the shell command
            result = subprocess.run(command, check=True, stdout=sys.stdout, stderr=sys.stderr)
            print(f"Synthetic data generated successfully. Output:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error in generating synthetic data: {e.stderr}")
            raise

    def _build_command_line_args(self, params):
        """Convert the parameters into a format that the external method can accept."""
        args = []
        for key, value in params.items():
            args.append(f"--{key}={value}")  # Format: --param=value
        return args

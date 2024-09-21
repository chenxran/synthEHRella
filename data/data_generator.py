import subprocess
import os


class DataGenerator:
    def __init__(self, method, num_gen_samples):
        """Initialize the DataGenerator with the path to the method folder."""
        self.method = method
        self.num_gen_samples = num_gen_samples

    def generate(self, params):
        """Run the external method to generate synthetic data."""
        generate_script = os.path.join("synthEHRella/data/scripts", f'gen_{self.method}.sh')  # Assuming it's a shell script
        if not os.path.exists(generate_script):
            raise FileNotFoundError(f"Generation script not found in {self.method_folder}")
        
        # Prepare command to execute the script with parameters if needed
        command = [generate_script] + " " + self._build_command_line_args(params)
        
        try:
            # Execute the shell command
            result = subprocess.run(command, check=True, capture_output=True, text=True)
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

# coding: utf-8
import io
import sys
from typing import List
import subprocess
import threading
import queue
import json

from langchain.tools import StructuredTool

from benchmark.src.tools.base import ToolFactory

PYTHON_REPL = "python_repl"

class PythonREPLSession:
    def __init__(self):
        self.process = subprocess.Popen(
            [sys.executable, "-i", "-u"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.output_queue = queue.Queue()
        self.is_reading = True

        self.stdout_thread = threading.Thread(target=self._read_output, args=(self.process.stdout,))
        self.stderr_thread = threading.Thread(target=self._read_output, args=(self.process.stderr,))
        self.stdout_thread.daemon = True
        self.stderr_thread.daemon = True
        self.stdout_thread.start()
        self.stderr_thread.start()

    def _read_output(self, pipe):
        while self.is_reading and not pipe.closed:
            try:
                line = pipe.readline()
                if line:
                    self.output_queue.put(line)
                else:
                    break
            except Exception:
                break
    
    def run(self, code: str, timeout: int = 10) -> str:
        """Execute Python code in the persistent REPL session."""
        if self.process.poll() is not None:
            return json.dumps({"stdout": "", "stderr": "Error: Python REPL session has been terminated."})
        
        # Unique markers to separate output
        end_marker = "f290d9e3-e298-4b53-b1c8-1c4b8b6a6b72"
        command = (
            "import sys, json\n"
            f"sys.stdout.write(json.dumps({{'output': repr({code})}}))\n"
            f"sys.stdout.flush()\n"
            f"print('{end_marker}')\n"
            "sys.stdout.flush()\n"
        )
        
        self.process.stdin.write(command)
        self.process.stdin.flush()

        output_buffer = []
        error_buffer = []
        
        try:
            while True:
                line = self.output_queue.get(timeout=timeout)
                if end_marker in line:
                    break
                output_buffer.append(line)
        except queue.Empty:
            return json.dumps({"stdout": "".join(output_buffer), "stderr": "Error: Command timed out."})
        
        # Process the captured output
        full_output = "".join(output_buffer)
        try:
            # The actual result is wrapped in JSON by our command
            result_json = json.loads(full_output)
            return json.dumps({"stdout": result_json.get('output', ''), "stderr": ""})
        except json.JSONDecodeError:
            # If JSON decoding fails, it's likely an error message printed directly
            return json.dumps({"stdout": "", "stderr": full_output})

    def close(self):
        """Close the Python REPL session."""
        self.is_reading = False
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
            except Exception:
                pass
            self.process.terminate()
            self.process.wait(timeout=2)
        
        self.stdout_thread.join(timeout=1)
        self.stderr_thread.join(timeout=1)


@ToolFactory.register(name=PYTHON_REPL, desc="A tool for executing Python code in a persistent REPL session.")
class PythonREPLTool:
    def __init__(self):
        self.session = PythonREPLSession()

    def get_tools(self) -> List[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=self.session.run,
                name="run_in_python_repl",
                description="Run Python code in the persistent REPL session.",
            )
        ]

    def __del__(self):
        if hasattr(self, 'session'):
            self.session.close() 
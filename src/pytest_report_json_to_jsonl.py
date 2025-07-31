import json

def json_to_jsonl(input_file_path, output_file_path):
    """
    Converts a JSON file (containing an array of objects) to a JSONL file.

    Args:
        input_file_path (str): Path to the input JSON file.
        output_file_path (str): Path to the output JSONL file.
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)

        # If this is a pytest-json-report format, extract the list of test entries
        if isinstance(data, dict) and 'tests' in data and isinstance(data['tests'], list):
            print(f"Info: extracting {len(data['tests'])} test records from report")
            data = data['tests']

        if not isinstance(data, list):
            print("Warning: Input JSON is not a list of objects. Each object will be written as a line.")
            # If it's a single object, wrap it in a list for consistent processing
            data = [data]

        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            for item in data:
                # Sum durations from setup, call, teardown if present
                total_duration = 0.0
                for phase in ("setup", "call", "teardown"):
                    if phase in item and isinstance(item[phase], dict):
                        total_duration += float(item[phase].get("duration", 0.0))
                item["duration"] = total_duration
                outfile.write(json.dumps(item) + '\n')
        print(f"Successfully converted '{input_file_path}' to '{output_file_path}'")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_file_path}'. Check file format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

json_to_jsonl("../tests/pytest_report.json", "../tests/pytest_report.jsonl")

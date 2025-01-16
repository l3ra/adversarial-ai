# Script to convert log file format
def convert_log_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Split the line into parts by tab
            parts = line.strip().split('\t')
            
            # Ensure there are 5 parts to process
            if len(parts) == 5:
                p_val = parts[0].strip()
                orig_sent = parts[1].strip()
                adv_sent = parts[2].strip()
                original_class = parts[3].strip()
                adversarial_class = parts[4].strip()
                
                # Format as a tuple
                formatted_line = f'   ({p_val}, "{orig_sent}", "{adv_sent}", {original_class}, {adversarial_class}),\n'
                outfile.write(formatted_line)
            else:
                print(f"Skipping malformed line: {line.strip()}")

# Specify input and output file paths
input_file_path = "record.log"   # Replace with the path to your input log file
output_file_path = "converted_log.txt"  # Replace with the desired output file path

# Call the function
convert_log_file(input_file_path, output_file_path)

print(f"Conversion complete. Output saved to {output_file_path}.")

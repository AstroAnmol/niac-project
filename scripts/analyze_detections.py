import os
import glob
import csv
import math
from collections import defaultdict

def analyze_latest_results(results_dir="/Users/sikka-mac/Research/Code/niac-project/Astrodynamics/Results"):
    # Find the latest detection_results file
    search_pattern = os.path.join(results_dir, "detection_results_*.csv")
    files = glob.glob(search_pattern)
    
    if not files:
        print("No detection results found in", results_dir)
        return
        
    latest_results_file = max(files, key=os.path.getmtime)
    print(f"Analyzing: {os.path.basename(latest_results_file)}")
    
    # Deriving the matching debris_samples file from the timestamp
    timestamp_part = latest_results_file.split('detection_results_')[-1]
    matching_samples_file = os.path.join(results_dir, f"debris_samples_{timestamp_part}")
    
    # Load debris samples into a dictionary for quick lookup by ID
    debris_states = {}
    if os.path.exists(matching_samples_file):
        with open(matching_samples_file, 'r') as f:
            reader = csv.reader(f)
            # Skip header
            next(reader)
            # Read satellite initial state (which is the first row after header)
            sat_row = next(reader)
            sat_r = [float(sat_row[0]), float(sat_row[1]), float(sat_row[2])]
            
            # Now read the debris samples, index 0 is debris ID 0
            for i, row in enumerate(reader):
                if len(row) >= 6:
                    r = [float(row[0]), float(row[1]), float(row[2])]
                    v = [float(row[3]), float(row[4]), float(row[5])]
                    rel_r = [r[0] - sat_r[0], r[1] - sat_r[1], r[2] - sat_r[2]]
                    sol_v = [1.2 * v[0], 1.2 * v[1], 1.2 * v[2]]
                    
                    debris_states[i] = {
                        'r': r,
                        'v': v,
                        'rel_r': rel_r,
                        'sol_v': sol_v
                    }
    else:
        print(f"Warning: Could not find matching samples file: {matching_samples_file}")

    interesting_debris = []
    total_detected = 0
    
    with open(latest_results_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['detected'].lower() == 'true':
                total_detected += 1
                debris_id = int(row['debris_id'])
                hits_str = row['sensor_hits']
                
                if not hits_str:
                    continue
                    
                hits = hits_str.split(';')
                sensors_hit = set()
                times_hit = []
                
                for hit in hits:
                    if '@' not in hit:
                        continue
                    sensor_part, time_part = hit.split('@')
                    sensors_hit.add(sensor_part)
                    times_hit.append(float(time_part))
                    
                # Check criteria: >1 sensor AND >1 timestamp
                if len(sensors_hit) > 1 and len(times_hit) > 1:
                    item_data = {
                        'debris_id': debris_id,
                        'num_sensors': len(sensors_hit),
                        'sensors': list(sensors_hit),
                        'num_times': len(set(times_hit)),
                        'hit_times': sorted(times_hit),
                        'first_detection': float(row['first_detection_time'])
                    }
                    if debris_id in debris_states:
                        item_data['state'] = debris_states[debris_id]
                    interesting_debris.append(item_data)
            
    # Print results to a file
    output_report_file = os.path.join(results_dir, f"analysis_report_{timestamp_part}.txt")
    with open(output_report_file, 'w') as outf:
        outf.write("=============================================\n")
        outf.write(f"Total Detected Debris: {total_detected}\n")
        outf.write(f"Debris meeting criteria (>1 sensor AND >1 timestamp): {len(interesting_debris)}\n")
        outf.write("=============================================\n\n")
        
        for item in interesting_debris:
            outf.write(f"Debris ID {item['debris_id']}:\n")
            if 'state' in item:
                r = item['state']['r']
                v = item['state']['v']
                rel_r = item['state']['rel_r']
                sol_v = item['state']['sol_v']
                outf.write(f"  - Position (km):       [{r[0]:.12f}, {r[1]:.12f}, {r[2]:.12f}]\n")
                outf.write(f"  - Velocity (km/s):     [{v[0]:.12f}, {v[1]:.12f}, {v[2]:.12f}]\n")
                outf.write(f"  - Rel Pos to Sat (km): [{rel_r[0]:.12f}, {rel_r[1]:.12f}, {rel_r[2]:.12f}]\n")
                sol_speed = math.sqrt(sol_v[0]**2 + sol_v[1]**2 + sol_v[2]**2)
                outf.write(f"  - Soliton Speed (km/s):  {sol_speed:.12f}\n")
            outf.write(f"  - Hit {item['num_sensors']} different sensors: {', '.join(item['sensors'])}\n")
            outf.write(f"  - Detected {item['num_times']} unique timestamps\n")
            outf.write(f"  - All Detection Times (s):\n")
            for t_hit in item['hit_times']:
                outf.write(f"      {t_hit:.12f}\n")
            outf.write("\n")
            
    print(f"Analysis saved to: {output_report_file}")

if __name__ == "__main__":
    analyze_latest_results()

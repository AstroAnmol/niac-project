import os
import glob
import csv
import math
from collections import defaultdict

def rv_to_oe(r, v):
    mu = 398600.4418
    r_mag = math.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
    v_mag = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    
    h = [
        r[1]*v[2] - r[2]*v[1],
        r[2]*v[0] - r[0]*v[2],
        r[0]*v[1] - r[1]*v[0]
    ]
    h_mag = math.sqrt(h[0]**2 + h[1]**2 + h[2]**2)
    n = [-h[1], h[0], 0]
    n_mag = math.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
    
    r_dot_v = r[0]*v[0] + r[1]*v[1] + r[2]*v[2]
    e_vec = [
        ((v_mag**2 - mu/r_mag)*r[0] - r_dot_v*v[0]) / mu,
        ((v_mag**2 - mu/r_mag)*r[1] - r_dot_v*v[1]) / mu,
        ((v_mag**2 - mu/r_mag)*r[2] - r_dot_v*v[2]) / mu
    ]
    e_mag = math.sqrt(e_vec[0]**2 + e_vec[1]**2 + e_vec[2]**2)
    
    epsilon = (v_mag**2)/2 - mu/r_mag
    if abs(epsilon) < 1e-12:
        a = float('inf')
    else:
        a = -mu / (2 * epsilon)
        
    inc = math.acos(max(-1.0, min(1.0, h[2] / h_mag))) if h_mag > 0 else 0
    
    if n_mag == 0:
        raan = 0
    else:
        raan = math.acos(max(-1.0, min(1.0, n[0] / n_mag)))
        if n[1] < 0:
            raan = 2*math.pi - raan
            
    if n_mag == 0 or e_mag == 0:
        arg_p = 0
    else:
        n_dot_e = n[0]*e_vec[0] + n[1]*e_vec[1] + n[2]*e_vec[2]
        arg_p = math.acos(max(-1.0, min(1.0, n_dot_e / (n_mag * e_mag))))
        if e_vec[2] < 0:
            arg_p = 2*math.pi - arg_p
            
    if e_mag == 0:
        ta = 0
    else:
        e_dot_r = e_vec[0]*r[0] + e_vec[1]*r[1] + e_vec[2]*r[2]
        ta = math.acos(max(-1.0, min(1.0, e_dot_r / (e_mag * r_mag))))
        if r_dot_v < 0:
            ta = 2*math.pi - ta
            
    return {
        'a': a, 'e': e_mag, 'i': math.degrees(inc),
        'raan': math.degrees(raan), 'arg_p': math.degrees(arg_p), 'ta': math.degrees(ta)
    }

def analyze_latest_results(target_dir=None):
    results_dir = "./Results"
    
    if target_dir is None:
        list_of_dirs = glob.glob(os.path.join(results_dir, 'Sim_*'))
        if not list_of_dirs:
            print(f"No Sim directories found in {results_dir}")
            return
        target_dir = max(list_of_dirs, key=os.path.getmtime)
    elif not os.path.isabs(target_dir) and not os.path.isdir(target_dir):
        potential_dir = os.path.join(results_dir, target_dir)
        if os.path.isdir(potential_dir):
            target_dir = potential_dir
            
    # Find the latest detection_results file in the target directory
    search_pattern = os.path.join(target_dir, "detection_results_*.csv")
    files = glob.glob(search_pattern)
    
    if not files:
        print("No detection results found in", target_dir)
        return
        
    latest_results_file = max(files, key=os.path.getmtime)
    print(f"Analyzing: {os.path.basename(latest_results_file)}")
    
    # Deriving the matching debris_samples file from the timestamp
    timestamp_part = latest_results_file.split('detection_results_')[-1]
    matching_samples_file = os.path.join(target_dir, f"debris_samples_{timestamp_part}")
    
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
    output_report_file = os.path.join(target_dir, f"analysis_report_{timestamp_part}.txt")
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
                
                oe = rv_to_oe(r, v)
                outf.write(f"  - Orbital Elements:\n")
                outf.write(f"      a (km): {oe['a']:.2f}\n")
                outf.write(f"      e:      {oe['e']:.6f}\n")
                outf.write(f"      i (deg): {oe['i']:.2f}\n")
                outf.write(f"      RAAN (deg): {oe['raan']:.2f}\n")
                outf.write(f"      ArgP (deg): {oe['arg_p']:.2f}\n")
                outf.write(f"      TA (deg): {oe['ta']:.2f}\n")
            outf.write(f"  - Hit {item['num_sensors']} different sensors: {', '.join(item['sensors'])}\n")
            outf.write(f"  - Detected {item['num_times']} unique timestamps\n")
            outf.write(f"  - All Detection Times (s):\n")
            for t_hit in item['hit_times']:
                outf.write(f"      {t_hit:.12f}\n")
            outf.write("\n")
            
    print(f"Analysis saved to: {output_report_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Analyze Detection Results.')
    parser.add_argument('--dir', type=str, help='Path to a target Sim_* directory.')
    args = parser.parse_args()
    
    analyze_latest_results(args.dir)

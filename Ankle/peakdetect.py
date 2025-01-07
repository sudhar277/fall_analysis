import subprocess
import sys
import pkg_resources
import os

def install_required_libraries():
    """
    Check and install required libraries if they're not already installed.
    Returns True if all installations are successful, False otherwise.
    """
    # List of required packages
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib'
    }
    
    # Get list of installed packages
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}
    
    # Check which packages need to be installed
    packages_to_install = []
    for package, pip_name in required_packages.items():
        if package not in installed_packages:
            packages_to_install.append(pip_name)
    
    # Install missing packages
    if packages_to_install:
        print("Installing required packages...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages_to_install)
            print("All required packages installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error installing packages: {e}")
            return False
    else:
        print("All required packages are already installed!")
        return True

# Now import the required libraries after installation
if install_required_libraries():
    import pandas as pd
    import numpy as np
    from scipy.signal import butter, filtfilt
    import matplotlib.pyplot as plt
    from pathlib import Path
else:
    print("Failed to install required packages. Please install them manually:")
    print("pip install pandas numpy scipy matplotlib")
    sys.exit(1)

def create_lowpass_filter(sampling_rate):
    """
    Create a Butterworth lowpass filter for smoothing the acceleration signal.
    """
    nyquist = sampling_rate / 2
    cutoff = 5  # 5 Hz cutoff frequency
    return butter(4, cutoff/nyquist, btype='low')

def process_single_file(file_path, time_window=1.0):
    """
    Process a single CSV file to detect and label the fall.
    """
    # Read the data
    data = pd.read_csv(file_path)
    
    # Calculate sampling rate
    data['time_diff'] = data['time'].diff().fillna(0)
    sampling_rate = 1 / data['time_diff'].mean()
    samples_window = int(time_window * sampling_rate)
    
    # Create and apply filter
    b, a = create_lowpass_filter(sampling_rate)
    
    # Calculate and filter acceleration magnitude
    data['acc_mag'] = np.sqrt(data['ax']**2 + data['ay']**2 + data['az']**2)
    data['acc_mag_filtered'] = filtfilt(b, a, data['acc_mag'])
    
    # Find the single highest peak
    peak_idx = data['acc_mag_filtered'].argmax()
    peak_time = data['time'].iloc[peak_idx]
    
    # Label the data
    data['label'] = 'non_fall'
    start_idx = max(0, peak_idx - samples_window)
    end_idx = min(len(data), peak_idx + samples_window)
    data.loc[start_idx:end_idx, 'label'] = 'backward_fall'
    
    return data, peak_idx, peak_time

def plot_and_save_results(data, peak_idx, output_path, filename):
    """
    Create and save visualization of the fall detection results.
    """
    plt.figure(figsize=(15, 8))
    
    plt.plot(data['time'], data['acc_mag_filtered'], 
             label='Filtered Acceleration', linewidth=2)
    
    plt.plot(data['time'].iloc[peak_idx], 
             data['acc_mag_filtered'].iloc[peak_idx], 
             'r*', markersize=15, 
             label='Fall Peak')
    
    fall_regions = data[data['label'] == 'backward_fall']
    for _, region in fall_regions.groupby((fall_regions.index != fall_regions.index - 1).cumsum()):
        plt.axvspan(region['time'].iloc[0], region['time'].iloc[-1], 
                    alpha=0.2, color='red', label='Fall Window')
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Acceleration Magnitude (m/s²)', fontsize=12)
    plt.title(f'Fall Detection Result - {filename}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    base_name = os.path.splitext(filename)[0]
    plot_path = os.path.join(output_path, f"{base_name}_plot.png")
    plt.savefig(plot_path)
    plt.close()

def process_folder(input_folder, output_folder):
    """
    Process all CSV files in the input folder and save results to output folder.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'plots'), exist_ok=True)
    
    # Create a summary DataFrame
    summary_data = []
    
    # Process each CSV file
    for file_path in Path(input_folder).glob('*.csv'):
        filename = file_path.name
        base_name = os.path.splitext(filename)[0]
        print(f"\nProcessing {filename}...")
        
        try:
            # Process the file
            labeled_data, peak_idx, peak_time = process_single_file(file_path)
            
            # Save labeled data
            output_path = os.path.join(output_folder, f"{base_name}_labeled.csv")
            labeled_data.to_csv(output_path, index=False)
            
            # Create and save plot
            plot_and_save_results(labeled_data, peak_idx, 
                                os.path.join(output_folder, 'plots'), 
                                filename)
            
            # Collect summary statistics
            summary_data.append({
                'filename': filename,
                'fall_peak_time': peak_time,
                'peak_acceleration': labeled_data['acc_mag_filtered'].iloc[peak_idx],
                'total_samples': len(labeled_data),
                'fall_samples': len(labeled_data[labeled_data['label'] == 'backward_fall']),
                'non_fall_samples': len(labeled_data[labeled_data['label'] == 'non_fall'])
            })
            
            print(f"Completed processing {filename}")
            print(f"Labeled data saved as: {base_name}_labeled.csv")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    if summary_data:
        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_folder, 'processing_summary.csv'), index=False)
        print("\nProcessing complete! Check the output folder for results.")
    else:
        print("\nNo files were successfully processed.")

if __name__ == "__main__":
    # Get input and output folders from user
    while True:
        input_folder = input("Enter the path to your input folder containing CSV files: ").strip()
        if os.path.exists(input_folder):
            break
        print("This folder doesn't exist. Please enter a valid path.")
    
    while True:
        output_folder = input("Enter the path where you want to save the results: ").strip()
        try:
            os.makedirs(output_folder, exist_ok=True)
            break
        except Exception as e:
            print(f"Error creating output folder: {e}")
            print("Please enter a valid path.")
    
    process_folder(input_folder, output_folder)
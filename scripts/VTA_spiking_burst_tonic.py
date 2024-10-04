# Exctract vurstand tonic portions fo the data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, norm, kstest, chi2_contingency, anderson
from scipy.optimize import minimize
from sklearn.decomposition import PCA
import seaborn as sns

# Step 1: Read the Data
def read_isi_data(file_path):
    """
    Reads the Excel file containing ISI data for multiple neurons.
    Returns a DataFrame of the ISI data.
    """
    # Load the ISI data from the Excel file
    isi_data = pd.read_excel(file_path)

    # Drop rows with missing values and return the cleaned DataFrame
    return isi_data

def scale_distribution_to_peak(pdf, data_peak, x_values):
    """
    Scale a distribution PDF so that its peak matches the peak of the corresponding data.
    Args:
        pdf (np.array): The PDF values of the distribution.
        data_peak (float): The peak value of the corresponding data.
        x_values (np.array): The x-axis values corresponding to the PDF.

    Returns:
        scaled_pdf (np.array): Scaled PDF so that its peak matches `data_peak`.
    """
    # Find the peak value of the PDF
    pdf_peak = np.max(pdf)

    # Calculate the scaling factor
    scaling_factor = data_peak / pdf_peak if pdf_peak > 0 else 1.0

    # Scale the PDF
    scaled_pdf = pdf * scaling_factor
    return scaled_pdf

def scale_to_match_combined(normalized_combined_pdf, scaled_component_pdf, x_values, region='burst'):
    """
    Scale the burst or tonic component such that it overlaps perfectly with the corresponding region of the normalized combined PDF.
    Args:
        normalized_combined_pdf (np.array): The combined normalized PDF.
        scaled_component_pdf (np.array): The scaled component PDF (burst or tonic).
        x_values (np.array): The x-axis values corresponding to the PDFs.
        region (str): 'burst' or 'tonic' to indicate which component is being scaled.

    Returns:
        scaled_component_pdf (np.array): The component PDF scaled to match the combined PDF.
    """
    # Determine the peak positions
    if region == 'burst':
        # Focus on the left part of the PDF for burst
        component_peak_index = np.argmax(scaled_component_pdf)  # Example: burst peak in first 0.05s
    else:
        # Focus on the right part of the PDF for tonic
        component_peak_index = np.argmax(scaled_component_pdf)  # Example: tonic peak after 0.05s

    # Find the peaks in the combined and component PDFs
    component_peak_value = scaled_component_pdf[component_peak_index]
    combined_peak_value = normalized_combined_pdf[component_peak_index]

    # Calculate the scaling factor
    scaling_factor = combined_peak_value / component_peak_value if component_peak_value > 0 else 1.0

    # Scale the component PDF using the scaling factor
    scaled_component_pdf = scaled_component_pdf * scaling_factor
    return scaled_component_pdf

def extract_parameters_for_cell(neuron_isis, x_values):
    """
    Extracts the specified parameters for a given cell's ISI data.
    """
    # Step 1: Fit log-normal to burst ISIs and normal to tonic ISIs
    burst_isis = neuron_isis[neuron_isis < 0.05]  # Example threshold for burst ISIs
    tonic_isis = neuron_isis[neuron_isis >= 0.05]  # Example threshold for tonic ISIs

    burst_shape, _, burst_scale = lognorm.fit(burst_isis, floc=0)
    tonic_mean, tonic_std = norm.fit(tonic_isis)

    # Calculate PDFs
    burst_pdf = lognorm.pdf(x_values, burst_shape, 0, burst_scale)
    tonic_pdf = norm.pdf(x_values, tonic_mean, tonic_std)

    # Create scaled versions of burst and tonic PDFs
    scaled_burst_pdf = burst_pdf.copy()
    scaled_tonic_pdf = tonic_pdf.copy()

    # Combine PDFs
    combined_pdf = scaled_burst_pdf + scaled_tonic_pdf
    normalized_combined_pdf = combined_pdf / np.trapz(combined_pdf, x_values)  # Normalize combined PDF

    # Step 2: Scale the burst and tonic components to match the combined PDF
    scaled_burst_pdf_matched = scale_to_match_combined(normalized_combined_pdf, scaled_burst_pdf, x_values, region='burst')
    scaled_tonic_pdf_matched = scale_to_match_combined(normalized_combined_pdf, scaled_tonic_pdf, x_values, region='tonic')

    # Step 3: Extract parameters from the matched PDFs
    burst_peak_value = np.max(scaled_burst_pdf_matched)  # Peak of burst component
    tonic_peak_value = np.max(scaled_tonic_pdf_matched)  # Peak of tonic component

    # Find the time (x-axis) positions of the peaks
    burst_peak_position = x_values[np.argmax(scaled_burst_pdf_matched)]
    tonic_peak_position = x_values[np.argmax(scaled_tonic_pdf_matched)]

    # Distance between burst and tonic peaks (in time)
    peak_distance = np.abs(burst_peak_position - tonic_peak_position)

    # Calculate AUCs
    auc_burst = np.trapz(scaled_burst_pdf_matched, x_values)
    auc_tonic = np.trapz(scaled_tonic_pdf_matched, x_values)

    return burst_peak_value, tonic_peak_value, peak_distance, auc_burst, auc_tonic


# Define path to the data file (adjust based on uploaded file location)
file_path = '/Users/reva/Documents/Neuron_SpikeTimes_BeforeCue_Concatenated.xlsx'

# Read ISI data
isi_data_full = read_isi_data(file_path)

# Prepare necessary variables for analysis
#x_values = np.linspace(0, 0.01, 1000)  # Range for x-axis values
bins = 1000  # Number of bins for histograms
valid_neurons_list = [col for col in isi_data_full.columns if not isi_data_full[col].dropna().empty]  # Filter valid neurons

# Store optimal parameters for all neurons
optimal_params_all_neurons_combined = {}
cell_parameters = []

# Process each neuron
for neuron in valid_neurons_list:
    spike_times = isi_data_full[neuron].dropna().values  # Extract ISI data
    neuron_isis = np.diff(spike_times)

    hist_values, bin_edges =np.histogram(neuron_isis, bins=bins, density=True)

    # Step 2: Calculate the bin centers from the bin edges
    x_values = (bin_edges[:-1] + bin_edges[1:]) / 2
    #plt.figure(figsize=(10, 6))
    #plt.hist(neuron_isis, bins=bins, density=True, alpha=0.6, color='blue', label='ISI Data')
    #plt.show()
    neuron_isis=neuron_isis[neuron_isis>0]
    burst_isis = neuron_isis[neuron_isis < 0.05]
    tonic_isis = neuron_isis[neuron_isis >= np.median(burst_isis)]
    # Store the optimal parameters
    burst_region = x_values[x_values < 0.05]  # Example: ISIs less than 0.05s are considered burst
    tonic_region = x_values[x_values >= np.median(burst_isis)]  # ISIs greater than 0.05s are considered tonic

    burst_peak_value = np.max(hist_values[x_values < 0.05])
    tonic_peak_value = np.max(hist_values[x_values >=  np.median(burst_isis)])
  
    burst_shape,_, burst_scale = lognorm.fit(burst_isis,floc=0)
    burst_params = (burst_shape, 0, burst_scale)
    
    tonic_mean ,loc, tonic_std = lognorm.fit(tonic_isis)
    tonic_params = (tonic_mean, loc, tonic_std)

    # Calculate PDFs
    x_values = np.linspace(0, np.max(neuron_isis), 1000)
    burst_pdf = lognorm.pdf(x_values, burst_shape, 0, burst_scale)
    tonic_pdf = lognorm.pdf(x_values, tonic_mean, loc, tonic_std)

    # Step 3: Scale each PDF to match the peak values
    scaled_burst_pdf = scale_distribution_to_peak(burst_pdf, burst_peak_value, x_values)
    scaled_tonic_pdf = scale_distribution_to_peak(tonic_pdf, tonic_peak_value, x_values)

    # Step 4: Combine the scaled PDFs
    combined_pdf = scaled_burst_pdf + scaled_tonic_pdf
    combined_pdf_ = combined_pdf/np.trapz(combined_pdf, x_values)
    scaled_burst_pdf_matched = scale_to_match_combined(combined_pdf_, scaled_burst_pdf, x_values, region='burst')

    # Step 3: Scale the tonic component to match the tonic region in the normalized combined PDF
    scaled_tonic_pdf_matched = scale_to_match_combined(combined_pdf_, scaled_tonic_pdf, x_values, region='tonic')

    # Step 5: Visualize the histogram and the combined PDF

    #plt.plot(x_values, scaled_burst_pdf, color='red', label='Scaled Burst PDF', linewidth=2)
    #plt.plot(x_values, combined_pdf, color='green', label='Scaled Tonic PDF', linewidth=2)
    #plt.plot(x_values, combined_pdf_, color='black', linestyle='--', label='Combined PDF', linewidth=2)
    #plt.fill_between(x_values, 0, scaled_burst_pdf_matched, color='red', alpha=0.5, label='Burst Component')
    #plt.fill_between(x_values, 0, scaled_tonic_pdf_matched, color='black', alpha=0.3, label='Tonic Component')

    #plt.xlabel('ISI (s)')
    #plt.ylabel('Density')
    #plt.title( f'ISI for Neuron: {neuron}')
    #plt.legend()
    #plt.grid()
    #plt.show()

    params = extract_parameters_for_cell(neuron_isis, x_values)
    cell_parameters.append(params)
# Display the optimal parameters for each neuron
#optimal_params_all_neurons_combined
columns = ['Burst Peak', 'Tonic Peak', 'Peak Distance (s)', 'AUC Burst', 'AUC Tonic']
df_cell_parameters = pd.DataFrame(cell_parameters, columns=columns)
plt.figure(figsize=(15, 8))
sns.set(style='whitegrid')

# Loop over each parameter and create a subplot for its swarm plot
for i, column in enumerate(df_cell_parameters.columns):
    plt.subplot(2, 3, i + 1)  # 2 rows, 3 columns for subplots
    sns.swarmplot(data=df_cell_parameters, y=column, color='blue')
    plt.title(f'Swarm Plot of {column}')
    plt.ylabel(column)
    plt.xlabel('Cells')

plt.tight_layout()
plt.show()
# Step 6: Perform PCA on the parameters
pca = PCA(n_components=2)  # Use 2 principal components for visualization
pca_result = pca.fit_transform(df_cell_parameters)

# Step 7: Visualize the PCA result
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], color='blue', alpha=0.7)
for i, (x, y) in enumerate(zip(pca_result[:, 0], pca_result[:, 1])):
    plt.text(x, y, f'Cell {i+1}', fontsize=9)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Cells Based on Extracted Parameters')
plt.grid()
plt.show()
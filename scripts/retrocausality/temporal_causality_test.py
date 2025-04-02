import numpy as np
import healpy as hp
from scipy import stats
import matplotlib.pyplot as plt
from astropy.io import fits
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

def temporal_causality_inversion_test(cmb_data, galaxy_data, n_simulations=1000):
    """
    Test whether CMB patterns exhibit correlations with future cosmic structures
    that exceed what conventional forward causality would predict.
    
    Parameters:
    -----------
    cmb_data : array-like
        The cosmic microwave background data (HEALPix map)
    galaxy_data : DataFrame
        Catalog of galaxies with positions and formation time estimates
    n_simulations : int
        Number of Monte Carlo simulations for statistical validation
    
    Returns:
    --------
    dict
        Results including temporal asymmetry metrics and statistical significance
    """
    import pandas as pd
    import time
    
    print("Starting Temporal Causality Inversion Test...")
    start_time = time.time()
    
    # Prep data
    nside = hp.get_nside(cmb_data)
    print(f"Analyzing HEALPix map with nside={nside}, containing {len(cmb_data)} pixels")
    
    # Convert galaxy positions to CMB coordinates
    theta, phi = galaxy_to_cmb_coordinates(galaxy_data['ra'], galaxy_data['dec'])
    
    # Extract features from the CMB around each galaxy position
    print("Extracting CMB features at galaxy positions...")
    features = []
    for i in range(len(galaxy_data)):
        features.append(extract_cmb_features(cmb_data, theta[i], phi[i], nside))
    features = np.array(features)
    
    # Get galaxy properties
    # Check which column names exist in the DataFrame
    mass_col = 'stellar_mass' if 'stellar_mass' in galaxy_data.columns else 'mass'
    metal_col = 'metallicity' if 'metallicity' in galaxy_data.columns else 'metal'
    
    galaxy = pd.DataFrame({
        'formation_time': galaxy_data['formation_time'],
        'mass': galaxy_data[mass_col],
        'metallicity': galaxy_data[metal_col],
    })
    
    # Step 2: Train model to predict galaxy properties from CMB patterns
    print("Training predictive model...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, galaxy, test_size=0.3, random_state=42
    )
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    predictions = model.predict(X_test)
    actual_r2 = r2_score(y_test, predictions)
    
    # Step 3: Generate surrogate data and evaluate prediction performance
    print(f"Generating {n_simulations} surrogate datasets for comparison...")
    
    surrogate_r2_scores = []
    
    for i in range(n_simulations):
        if i % 100 == 0 and i > 0:
            print(f"Completed {i}/{n_simulations} simulations")
        
        # Generate surrogate CMB with same power spectrum but randomized phases
        surrogate_map = generate_surrogate_map(cmb_data)
        
        # Extract features from surrogate at same galaxy positions
        surrogate_features = []
        for j in range(len(galaxy_data)):
            surrogate_features.append(extract_cmb_features(surrogate_map, theta[j], phi[j], nside))
        
        surrogate_features = np.array(surrogate_features)
        
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            surrogate_features, galaxy, test_size=0.3, random_state=42
        )
        
        # Train and evaluate model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        surrogate_r2 = r2_score(y_test, predictions)
        surrogate_r2_scores.append(surrogate_r2)
    
    # Calculate statistical significance
    surrogate_mean = np.mean(surrogate_r2_scores)
    surrogate_std = np.std(surrogate_r2_scores)
    
    z_score = (actual_r2 - surrogate_mean) / surrogate_std if surrogate_std > 0 else 0
    p_value = 1 - stats.norm.cdf(z_score) if surrogate_std > 0 else 1.0

    # Step 4: Test for temporal asymmetry
    print("Testing for temporal asymmetry...")
    
    # Split galaxies by formation time
    median_formation_time = np.median(galaxy_data['formation_time'])
    early_indices = np.where(galaxy_data['formation_time'] < median_formation_time)[0]
    late_indices = np.where(galaxy_data['formation_time'] >= median_formation_time)[0]
    
    # Extract features for early and late forming galaxies
    early_features = features[early_indices]
    early_properties = galaxy.iloc[early_indices]
    
    late_features = features[late_indices]
    late_properties = galaxy.iloc[late_indices]
    
    # Evaluate prediction model for each group
    early_r2 = evaluate_prediction_model(early_features, early_properties)
    late_r2 = evaluate_prediction_model(late_features, late_properties)
    
    # Calculate temporal asymmetry score (positive = better prediction for late-forming galaxies)
    temporal_asymmetry = late_r2 - early_r2
    
    # Get statistical significance of temporal asymmetry
    # Run similar analysis on surrogate data
    surrogate_asymmetries = []
    
    for i in range(min(100, n_simulations)):  # Use subset for efficiency
        # Use already generated surrogate features from previous step
        surrogate_early_features = surrogate_features[early_indices]
        surrogate_early_properties = galaxy.iloc[early_indices]
        
        surrogate_late_features = surrogate_features[late_indices]
        surrogate_late_properties = galaxy.iloc[late_indices]
        
        # Evaluate
        surrogate_early_r2 = evaluate_prediction_model(
            surrogate_early_features, surrogate_early_properties
        )
        surrogate_late_r2 = evaluate_prediction_model(
            surrogate_late_features, surrogate_late_properties
        )
        
        surrogate_asymmetry = surrogate_late_r2 - surrogate_early_r2
        surrogate_asymmetries.append(surrogate_asymmetry)
    
    asymmetry_mean = np.mean(surrogate_asymmetries)
    asymmetry_std = np.std(surrogate_asymmetries)
    
    asymmetry_z = (temporal_asymmetry - asymmetry_mean) / asymmetry_std if asymmetry_std > 0 else 0
    asymmetry_p = 1 - stats.norm.cdf(asymmetry_z) if asymmetry_std > 0 else 1.0
    
    # Step 5: Analyze phi-optimization
    print("Analyzing phi-related formation time specialization...")
    
    # Determine which galaxies have formation times related to golden ratio
    if 'is_phi_related' in galaxy_data.columns:
        galaxy_data['phi_related_formation'] = galaxy_data['is_phi_related']
    else:
        galaxy_data['phi_related_formation'] = is_phi_related_time(galaxy_data['formation_time'])
    
    # Split galaxies by phi-relatedness
    phi_indices = np.where(galaxy_data['phi_related_formation'] == 1)[0]
    non_phi_indices = np.where(galaxy_data['phi_related_formation'] == 0)[0]
    
    # Extract features
    phi_features = features[phi_indices]
    phi_properties = galaxy.iloc[phi_indices]
    
    non_phi_features = features[non_phi_indices]
    non_phi_properties = galaxy.iloc[non_phi_indices]
    
    # Evaluate prediction model for each group
    phi_r2 = evaluate_prediction_model(phi_features, phi_properties)
    non_phi_r2 = evaluate_prediction_model(non_phi_features, non_phi_properties)
    
    # Calculate phi optimization score
    phi_optimization = phi_r2 - non_phi_r2
    
    # Get statistical significance of phi optimization
    surrogate_phi_opts = []
    
    for i in range(min(100, n_simulations)):
        surrogate_phi_features = surrogate_features[phi_indices]
        surrogate_phi_properties = galaxy.iloc[phi_indices]
        
        surrogate_non_phi_features = surrogate_features[non_phi_indices]
        surrogate_non_phi_properties = galaxy.iloc[non_phi_indices]
        
        surrogate_phi_r2 = evaluate_prediction_model(
            surrogate_phi_features, surrogate_phi_properties
        )
        surrogate_non_phi_r2 = evaluate_prediction_model(
            surrogate_non_phi_features, surrogate_non_phi_properties
        )
        
        surrogate_phi_opt = surrogate_phi_r2 - surrogate_non_phi_r2
        surrogate_phi_opts.append(surrogate_phi_opt)
    
    phi_opt_mean = np.mean(surrogate_phi_opts)
    phi_opt_std = np.std(surrogate_phi_opts)
    
    phi_opt_z = (phi_optimization - phi_opt_mean) / phi_opt_std if phi_opt_std > 0 else 0
    phi_opt_p = 1 - stats.norm.cdf(phi_opt_z) if phi_opt_std > 0 else 1.0
    
    # Log all results
    print("\n" + "="*50)
    print("TEMPORAL CAUSALITY INVERSION TEST RESULTS")
    print("="*50)
    
    print(f"\nPrediction of Galaxy Properties from CMB Patterns:")
    print(f"CMB Data R² Score: {actual_r2:.4f}")
    print(f"Surrogate Data R² Score: {surrogate_mean:.4f}")
    print(f"Ratio: {actual_r2/surrogate_mean:.2f}x")
    print(f"Z-score: {z_score:.4f}")
    print(f"P-value: {p_value:.8f}")
    
    print(f"\nTemporal Asymmetry Analysis:")
    print(f"Early-forming Galaxies R²: {early_r2:.4f}")
    print(f"Late-forming Galaxies R²: {late_r2:.4f}")
    print(f"Temporal Asymmetry: {temporal_asymmetry:.4f}")
    print(f"Surrogate Asymmetry: {asymmetry_mean:.4f}")
    print(f"Z-score: {asymmetry_z:.4f}")
    print(f"P-value: {asymmetry_p:.8f}")
    
    print(f"\nPhi-Optimization in Temporal Relationships:")
    print(f"Phi-related Formation Time R²: {phi_r2:.4f}")
    print(f"Non-Phi-related Formation Time R²: {non_phi_r2:.4f}")
    print(f"Phi Optimization: {phi_optimization:.4f}")
    print(f"Surrogate Phi Optimization: {phi_opt_mean:.4f}")
    print(f"Z-score: {phi_opt_z:.4f}")
    print(f"P-value: {phi_opt_p:.8f}")
    
    execution_time = time.time() - start_time
    print(f"\nTest completed in {execution_time:.2f} seconds")
    
    # Return comprehensive results
    results = {
        "actual_r2": actual_r2,
        "surrogate_mean": surrogate_mean,
        "surrogate_std": surrogate_std,
        "z_score": z_score,
        "p_value": p_value,
        "early_r2": early_r2,
        "late_r2": late_r2,
        "temporal_asymmetry": temporal_asymmetry,
        "asymmetry_mean": asymmetry_mean,
        "asymmetry_std": asymmetry_std,
        "asymmetry_z": asymmetry_z,
        "asymmetry_p": asymmetry_p,
        "phi_r2": phi_r2,
        "non_phi_r2": non_phi_r2,
        "phi_optimization": phi_optimization,
        "phi_opt_mean": phi_opt_mean,
        "phi_opt_std": phi_opt_std,
        "phi_opt_z": phi_opt_z,
        "phi_opt_p": phi_opt_p,
        "execution_time": execution_time
    }
    
    return results

def galaxy_to_cmb_coordinates(ra, dec):
    """Convert galaxy sky coordinates to CMB spherical coordinates"""
    # Convert from degrees to radians
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    
    # Convert to HEALPix theta/phi convention
    # Note: theta is colatitude (0 = North Pole, pi = South Pole)
    theta = np.pi/2 - dec_rad
    phi = ra_rad
    
    return theta, phi

def extract_cmb_features(cmb_data, theta, phi, nside, radius_arcmin=30):
    """
    Extract features from the CMB map around a given position.
    
    Parameters:
    -----------
    cmb_data : array-like
        HEALPix map of CMB data
    theta, phi : float
        Spherical coordinates of the position
    nside : int
        HEALPix nside parameter
    radius_arcmin : float
        Radius in arcminutes to extract features
    
    Returns:
    --------
    features : array
        Vector of extracted features
    """
    # Constants
    PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    # Convert position to pixel index
    pix = hp.ang2pix(nside, theta, phi)
    
    # Convert radius from arcminutes to radians
    radius_rad = np.radians(radius_arcmin / 60.0)
    
    # Get pixels within radius
    vec = hp.pix2vec(nside, pix)
    pixels = hp.query_disc(nside, vec, radius_rad)
    
    # Extract CMB values
    values = cmb_data[pixels]
    
    # Calculate statistical features
    features = [
        np.mean(values),                # Mean temperature
        np.std(values),                 # Standard deviation
        stats.skew(values),             # Skewness
        stats.kurtosis(values),         # Kurtosis
        np.percentile(values, 10),      # 10th percentile
        np.percentile(values, 25),      # 25th percentile
        np.percentile(values, 50),      # Median
        np.percentile(values, 75),      # 75th percentile
        np.percentile(values, 90),      # 90th percentile
        np.max(values) - np.min(values) # Range
    ]
    
    # Add wavelet features (simple approximation)
    # Divide the disc into concentric rings
    center_value = cmb_data[pix]
    ring_features = []
    
    ring_radii = [radius_rad * r for r in [0.2, 0.4, 0.6, 0.8, 1.0]]
    prev_ring_pixels = []
    
    for i, radius in enumerate(ring_radii):
        ring_pixels = hp.query_disc(nside, vec, radius)
        
        if i > 0:
            # Get only pixels in this ring (not including inner rings)
            ring_only = np.setdiff1d(ring_pixels, prev_ring_pixels)
            if len(ring_only) > 0:
                ring_values = cmb_data[ring_only]
                ring_features.extend([
                    np.mean(ring_values),
                    np.std(ring_values)
                ])
            else:
                ring_features.extend([0, 0])
        
        prev_ring_pixels = ring_pixels
    
    features.extend(ring_features)
    
    # Add phi-derived features
    phi_ratio = PHI
    phi_related_features = []
    
    # Find pixels at golden ratio distances
    for i in range(1, 4):
        phi_radius = radius_rad * (phi_ratio ** i)
        if phi_radius < np.pi:  # Ensure radius is valid
            phi_pixels = hp.query_disc(nside, vec, phi_radius)
            phi_only = np.setdiff1d(phi_pixels, prev_ring_pixels)
            
            if len(phi_only) > 0:
                phi_values = cmb_data[phi_only]
                phi_related_features.extend([
                    np.mean(phi_values),
                    np.std(phi_values),
                    np.mean(phi_values) - center_value  # Relation to center
                ])
            else:
                phi_related_features.extend([0, 0, 0])
    
    features.extend(phi_related_features)
    
    return np.array(features)

def generate_surrogate_map(cmb_data):
    """Generate a surrogate map with the same power spectrum but randomized phases"""
    # Get map properties
    nside = hp.get_nside(cmb_data)
    
    # Convert to alm (spherical harmonic coefficients)
    lmax = 3 * nside - 1
    alm = hp.map2alm(cmb_data, lmax=lmax)
    
    # Randomize phases
    randomized_alm = randomize_phases(alm)
    
    # Convert back to map
    surrogate_map = hp.alm2map(randomized_alm, nside)
    
    return surrogate_map

def randomize_phases(alm):
    """Randomize phases of spherical harmonic coefficients"""
    lmax = hp.Alm.getlmax(len(alm))
    randomized_alm = np.zeros_like(alm)
    
    for l in range(lmax + 1):
        for m in range(l + 1):
            idx = hp.Alm.getidx(lmax, l, m)
            amp = np.abs(alm[idx])
            phase = np.random.uniform(0, 2*np.pi)
            randomized_alm[idx] = amp * np.exp(1j * phase)
    
    return randomized_alm

def evaluate_prediction_model(features, properties, test_size=0.3):
    """Train and evaluate a model to predict galaxy properties from CMB features"""
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            features, properties, test_size=test_size, random_state=42
        )
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        
        return r2
    except:
        # Return 0 if insufficient data or other issues
        return 0.0

def is_phi_related_time(times, tolerance=0.05):
    """
    Determine whether formation times are related to the golden ratio.
    
    Parameters:
    -----------
    times : array-like
        Formation times in Gyr since Big Bang
    tolerance : float
        Tolerance for considering a time phi-related
    
    Returns:
    --------
    is_phi_related : array
        Binary array indicating whether each time is phi-related
    """
    is_phi_related = np.zeros_like(times, dtype=int)
    PHI = (1 + np.sqrt(5)) / 2
    
    # Universe age is approximately 13.8 Gyr
    universe_age = 13.8
    
    # Check if time is related to phi^n * universe_age for some n
    for n in range(-5, 6):
        phi_time = universe_age / (PHI ** n)
        
        # Time is phi-related if within tolerance of phi^n * universe_age
        is_phi_related |= (np.abs(times - phi_time) / phi_time < tolerance)
    
    return is_phi_related

def visualize_results(results):
    """Create visualizations of temporal causality test results"""
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Prediction accuracy comparison
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.bar(['Actual CMB', 'Surrogate'], 
           [results['actual_r2'], results['surrogate_mean']], 
           yerr=[0, results['surrogate_std']],
           color=['blue', 'gray'])
    ax1.set_ylabel('R² Score')
    ax1.set_title('Prediction Accuracy from CMB to Galaxy Properties')
    
    # Add statistical information
    stats_text = "Z-score: {:.4f}\nP-value: {:.8f}".format(
        results['z_score'], results['p_value'])
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, va='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Temporal asymmetry visualization
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.bar(['Early-forming\nGalaxies', 'Late-forming\nGalaxies'], 
           [results['early_r2'], results['late_r2']])
    ax2.set_ylabel('R² Score')
    ax2.set_title('Temporal Asymmetry in Prediction Accuracy')
    
    # Add asymmetry information
    asymm_text = "Temporal Asymmetry: {:.4f}\nZ-score: {:.4f}\nP-value: {:.8f}".format(
        results['temporal_asymmetry'], results['asymmetry_z'], results['asymmetry_p'])
    ax2.text(0.05, 0.95, asymm_text, transform=ax2.transAxes, va='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Phi optimization visualization
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.bar(['Phi-related\nFormation Times', 'Non-Phi-related\nFormation Times'], 
           [results['phi_r2'], results['non_phi_r2']])
    ax3.set_ylabel('R² Score')
    ax3.set_title('Phi Optimization in Temporal Relationships')
    
    # Add phi optimization information
    phi_text = "Phi Optimization: {:.4f}\nZ-score: {:.4f}\nP-value: {:.8f}".format(
        results['phi_optimization'], results['phi_opt_z'], results['phi_opt_p'])
    ax3.text(0.05, 0.95, phi_text, transform=ax3.transAxes, va='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Summary gauge
    ax4 = fig.add_subplot(2, 2, 4)
    create_summary_gauge(ax4, results)
    
    plt.tight_layout()
    return fig

def create_summary_gauge(ax, results):
    """Create a gauge visualization summarizing temporal causality findings"""
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    # Draw gauge background
    theta = np.linspace(-np.pi, 0, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    ax.plot(x, y, 'k-', linewidth=2)
    
    # Add colored regions
    theta_classical = np.linspace(-np.pi, -2*np.pi/3, 50)
    x_classical = np.cos(theta_classical)
    y_classical = np.sin(theta_classical)
    ax.fill_between(x_classical, 0, y_classical, color='blue', alpha=0.3)
    
    theta_quantum = np.linspace(-2*np.pi/3, -np.pi/3, 50)
    x_quantum = np.cos(theta_quantum)
    y_quantum = np.sin(theta_quantum)
    ax.fill_between(x_quantum, 0, y_quantum, color='purple', alpha=0.3)
    
    theta_beyond = np.linspace(-np.pi/3, 0, 50)
    x_beyond = np.cos(theta_beyond)
    y_beyond = np.sin(theta_beyond)
    ax.fill_between(x_beyond, 0, y_beyond, color='red', alpha=0.3)
    
    # Add labels
    ax.text(-0.9, -0.3, "Forward\nCausality", ha='center', va='center')
    ax.text(0, -0.3, "Quantum\nWeirdness", ha='center', va='center')
    ax.text(0.9, -0.3, "Retro-\ncausality", ha='center', va='center')
    
    # Get z-score (combine asymmetry and phi optimization)
    combined_z = max(results['asymmetry_z'], results['phi_opt_z'])
    combined_z = min(max(combined_z, 0), 10)  # Clamp to [0, 10]
    
    # Map z-score to angle
    angle = -np.pi * (1 - combined_z/10)
    
    # Draw gauge needle
    needle_x = 0.8 * np.cos(angle)
    needle_y = 0.8 * np.sin(angle)
    ax.plot([0, needle_x], [0, needle_y], 'k-', linewidth=3)
    ax.plot(0, 0, 'ko', markersize=10)
    
    # Add key metrics
    metrics_text = "Temporal Asymmetry: {:.4f}\nZ-score: {:.2f}\n".format(
        results['temporal_asymmetry'], results['asymmetry_z'])
    metrics_text += "Phi Optimization: {:.4f}\nZ-score: {:.2f}".format(
        results['phi_optimization'], results['phi_opt_z'])
    
    ax.text(0, -0.6, metrics_text, ha='center', va='center', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add interpretation
    if (results['asymmetry_p'] < 0.01 and results['phi_opt_p'] < 0.05) or \
       (results['asymmetry_p'] < 0.05 and results['phi_opt_p'] < 0.01):
        interpretation = "STRONG EVIDENCE for retrocausality"
    elif results['asymmetry_p'] < 0.05 or results['phi_opt_p'] < 0.05:
        interpretation = "MODERATE EVIDENCE for retrocausality"
    elif results['asymmetry_p'] < 0.1 or results['phi_opt_p'] < 0.1:
        interpretation = "WEAK EVIDENCE for retrocausality"
    else:
        interpretation = "NO SIGNIFICANT EVIDENCE for retrocausality"
        
    ax.text(0, -0.9, interpretation, ha='center', va='center', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Remove ticks and spines
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    ax.set_title('Temporal Causality Evidence Gauge')

def load_cmb_data(filename):
    """
    Load CMB data from a FITS file or text file.
    
    Parameters:
    -----------
    filename : str
        Path to CMB data file (FITS or text)
    
    Returns:
    --------
    cmb_data : array-like
        HEALPix map of CMB data
    """
    import os
    from astropy.io import fits
    import healpy as hp
    import numpy as np
    
    print(f"Loading CMB data from {filename}...")
    
    file_extension = os.path.splitext(filename)[1].lower()
    
    if file_extension == '.fits':
        try:
            # First attempt: standard HEALPix map
            cmb_data = hp.read_map(filename)
            print(f"Successfully loaded HEALPix map with nside={hp.get_nside(cmb_data)}")
            return cmb_data
        except Exception as e:
            print(f"Standard HEALPix loading failed: {str(e)}")
            try:
                # Second attempt: load as standard FITS file
                with fits.open(filename) as hdul:
                    # Try to determine which HDU contains the data
                    for i, hdu in enumerate(hdul):
                        if hasattr(hdu, 'data') and hdu.data is not None:
                            if len(hdu.data.shape) > 0:  # Skip empty HDUs
                                print(f"Found data in HDU {i}")
                                # Convert to HEALPix format if needed
                                data = np.array(hdu.data)
                                if len(data.shape) == 2:
                                    # If it's a 2D image, we need to convert to HEALPix
                                    print("Converting 2D image to HEALPix...")
                                    nside = 1024  # Default nside
                                    cmb_data = np.zeros(hp.nside2npix(nside))
                                    # Simple conversion (this is a simplification)
                                    for j in range(min(len(cmb_data), len(data.flatten()))):
                                        cmb_data[j] = data.flatten()[j]
                                else:
                                    cmb_data = data
                                print(f"Data shape: {cmb_data.shape}")
                                return cmb_data
                    
                    # If we reach here, no suitable data was found
                    raise ValueError("No usable data found in FITS file")
            except Exception as e:
                print(f"FITS loading failed: {str(e)}")
                
    elif file_extension == '.txt':
        try:
            # For text files containing power spectrum (like planck_tt_spectrum_*.txt)
            # We'll convert the power spectrum to a simplified map
            print("Loading power spectrum from text file...")
            data = np.loadtxt(filename)
            
            # Extract multipoles and corresponding power
            if data.shape[1] >= 2:
                ell = data[:, 0].astype(int)
                cl = data[:, 1]
                
                # Fill in missing multipoles if needed
                max_ell = max(ell)
                full_cl = np.zeros(max_ell + 1)
                full_cl[ell] = cl
                
                # Generate a random map with this power spectrum
                print("Generating HEALPix map from power spectrum...")
                nside = 256  # Reasonable resolution
                cmb_data = hp.synfast(full_cl, nside)
                print(f"Created synthetic map with nside={nside}")
                return cmb_data
            else:
                raise ValueError("Text file doesn't have expected column structure")
        except Exception as e:
            print(f"Text file loading failed: {str(e)}")
    
    # If all methods fail
    print("WARNING: Generating random CMB data for testing")
    nside = 256
    cmb_data = hp.synfast(np.ones(1000), nside)
    return cmb_data

def load_galaxy_data(filename):
    """Load galaxy catalog data from a CSV or FITS file"""
    try:
        # Try as CSV first
        import pandas as pd
        galaxy_data = pd.read_csv(filename)
        
        # Check for required columns
        required_columns = ['ra', 'dec', 'formation_time', 'mass', 'metallicity']
        missing_columns = [col for col in required_columns if col not in galaxy_data.columns]
        
        if missing_columns:
            print(f"Warning: Missing required columns: {missing_columns}")
            print("Attempting to use alternative column names...")
            
            # Try alternative column names
            alt_columns = {
                'ra': ['RA', 'right_ascension', 'alpha', 'ALPHA'],
                'dec': ['DEC', 'declination', 'delta', 'DELTA'],
                'formation_time': ['formation', 'age', 'formation_age', 'form_time'],
                'mass': ['MASS', 'm_star', 'stellar_mass', 'M'],
                'metallicity': ['metal', 'Z', 'z_met', 'MET']
            }
            
            for req_col, alt_cols in alt_columns.items():
                if req_col in missing_columns:
                    for alt_col in alt_cols:
                        if alt_col in galaxy_data.columns:
                            galaxy_data[req_col] = galaxy_data[alt_col]
                            print(f"Using '{alt_col}' as '{req_col}'")
                            break
        
        # Check again for required columns
        missing_columns = [col for col in required_columns if col not in galaxy_data.columns]
        
        if missing_columns:
            print(f"Error: Still missing required columns: {missing_columns}")
            print("Please ensure your galaxy catalog has the required data:")
            print("  - ra: Right ascension in degrees")
            print("  - dec: Declination in degrees")
            print("  - formation_time: Galaxy formation time in Gyr since Big Bang")
            print("  - mass: Galaxy mass (typically in solar masses)")
            print("  - metallicity: Galaxy metallicity")
            return None
        
        # If formation_time isn't available but redshift is, convert
        if 'formation_time' in missing_columns and 'redshift' in galaxy_data.columns:
            print("Converting redshift to formation time...")
            galaxy_data['formation_time'] = redshift_to_formation_time(galaxy_data['redshift'])
        
        return galaxy_data
    
    except Exception as e:
        print(f"Error loading galaxy data: {e}")
        try:
            # Try as FITS
            hdulist = fits.open(filename)
            data = hdulist[1].data
            hdulist.close()
            
            # Convert to DataFrame
            galaxy_data = pd.DataFrame(data)
            
            # Check for required columns (similar logic as above)
            # ...
            
            return galaxy_data
        except:
            print(f"Could not load {filename} as CSV or FITS. Please check the file format.")
            return None

def redshift_to_formation_time(redshift, H0=67.8, Om0=0.308):
    """
    Convert redshift to formation time in Gyr since the Big Bang.
    
    Parameters:
    -----------
    redshift : array-like
        Galaxy redshift values
    H0 : float
        Hubble constant in km/s/Mpc
    Om0 : float
        Matter density parameter
    
    Returns:
    --------
    formation_time : array-like
        Formation time in Gyr since the Big Bang
    """
    try:
        from astropy.cosmology import FlatLambdaCDM
        import astropy.units as u
        
        # Create cosmology object
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        
        # Convert redshift to age
        age_at_z = cosmo.age(redshift).to(u.Gyr).value
        
        # Subtract from the current age of the universe
        universe_age = cosmo.age(0).to(u.Gyr).value
        formation_time = universe_age - age_at_z
        
        return formation_time
    except ImportError:
        print("Warning: astropy not available for accurate conversion.")
        # Simplified approximation (less accurate)
        return 13.8 - 13.8 * (1 / (1 + redshift))

def create_simulated_galaxy_data(n_galaxies=10000):
    """
    Create simulated galaxy data if real data is not available.
    This is for TESTING PURPOSES ONLY.
    
    Parameters:
    -----------
    n_galaxies : int
        Number of galaxies to simulate
    
    Returns:
    --------
    galaxy_data : DataFrame
        Simulated galaxy catalog
    """
    import pandas as pd
    import numpy as np
    
    print("Creating simulated galaxy data for testing...")
    print("WARNING: This is for code testing only and will not produce meaningful scientific results.")
    
    # Generate random sky positions
    ra = np.random.uniform(0, 360, n_galaxies)
    dec = np.random.uniform(-90, 90, n_galaxies)
    
    # Generate formation times (biased toward earlier universe)
    formation_time = np.random.beta(2, 5, n_galaxies) * 13.8  # in Gyr
    
    # Generate masses (log-normal distribution)
    log_mass = np.random.normal(10, 1, n_galaxies)  # log solar masses
    mass = 10**log_mass
    
    # Generate metallicities (correlated with mass and formation time)
    z_base = np.random.normal(0, 1, n_galaxies)
    metallicity = 0.02 * (mass / 1e10)**0.3 * (1 - formation_time/13.8)**0.5 * (0.5 + 0.5 * z_base)
    metallicity = np.clip(metallicity, 0.0001, 0.05)
    
    # Create DataFrame
    galaxy_data = pd.DataFrame({
        'ra': ra,
        'dec': dec,
        'formation_time': formation_time,
        'mass': mass,
        'metallicity': metallicity
    })
    
    return galaxy_data

def create_scale55_enhanced_galaxy_data(cmb_data, n_galaxies=10000, phi_bias=0.3):
    """
    Create simulated galaxy data with enhanced correlation to scale 55 CMB features,
    specifically optimized around the Golden Ratio.
    
    Parameters:
    -----------
    cmb_data : array-like
        HEALPix map of CMB data to extract scale 55 features from
    n_galaxies : int
        Number of galaxies to simulate
    phi_bias : float
        Strength of bias towards Golden Ratio formation times (0-1)
    
    Returns:
    --------
    galaxy_data : DataFrame
        Simulated galaxy catalog with enhanced scale 55 correlations
    """
    import pandas as pd
    import healpy as hp
    from scipy import stats
    
    # Calculate CMB power spectrum
    nside = hp.get_nside(cmb_data)
    alm = hp.map2alm(cmb_data)
    cl = hp.alm2cl(alm)
    
    # Extract scale 55 power and features
    scale_55_power = cl[55] if len(cl) > 55 else cl[-1]
    
    # Uniform random positions on the sphere
    phi = np.random.uniform(0, 2*np.pi, size=n_galaxies)
    theta = np.arccos(np.random.uniform(-1, 1, size=n_galaxies))
    
    # Convert to RA, DEC
    ra = np.rad2deg(phi)
    dec = 90 - np.rad2deg(theta)
    
    # Get pixel indices for each position
    pixels = hp.ang2pix(nside, theta, phi)
    
    # Extract CMB temperature at each position
    cmb_temp = cmb_data[pixels]
    
    # Extract scale 55 features
    scale_features = []
    for i in range(n_galaxies):
        # Get features around each galaxy position with emphasis on scale 55
        features = extract_cmb_features(cmb_data, theta[i], phi[i], nside, radius_arcmin=30)
        
        # Calculate specific scale 55 feature (simplified approximation)
        # In a real implementation, this would use proper spherical harmonic decomposition
        scale_features.append(np.mean(features))
    
    scale_features = np.array(scale_features)
    
    # Define Golden Ratio
    PHI = (1 + np.sqrt(5)) / 2
    
    # Base redshifts from a realistic distribution
    # This creates a distribution peaked around z=2 with tail extending to z=6
    redshifts = np.random.lognormal(mean=0.7, sigma=0.5, size=n_galaxies)
    
    # Convert redshifts to formation times
    formation_times = redshift_to_formation_time(redshifts)
    
    # Create phi-related formation times biased by scale 55 features
    # Universe age is approximately 13.8 Gyr
    universe_age = 13.8
    
    # Calculate phi-related time benchmarks
    phi_times = []
    for n in range(-5, 6):
        phi_times.append(universe_age / (PHI ** n))
    
    # Adjust some formation times to be closer to phi-related times
    # The higher the scale 55 feature, the more likely to be phi-related
    scale_features_normalized = (scale_features - np.mean(scale_features)) / np.std(scale_features)
    
    for i in range(n_galaxies):
        # Probability of being adjusted towards a phi-related time
        if np.random.random() < phi_bias * (1 + scale_features_normalized[i]):
            # Pick a random phi-related time
            target_time = np.random.choice(phi_times)
            # Adjust the formation time partially towards the phi-related time
            formation_times[i] = formation_times[i] + 0.6 * (target_time - formation_times[i])
    
    # Create other galaxy properties with correlations to scale 55 features
    stellar_mass = 10**np.random.normal(10, 1, size=n_galaxies)
    stellar_mass *= (1 + 0.2 * scale_features_normalized)  # Correlate with scale 55
    
    metallicity = np.random.normal(-0.5, 0.3, size=n_galaxies)
    metallicity += 0.15 * scale_features_normalized  # Correlate with scale 55
    
    # Create DataFrame
    galaxy_data = pd.DataFrame({
        'ra': ra,
        'dec': dec,
        'redshift': redshifts,
        'formation_time': formation_times,
        'stellar_mass': stellar_mass,
        'metallicity': metallicity,
        'is_phi_related': is_phi_related_time(formation_times)
    })
    
    return galaxy_data

# Main function
def main(cmb_data, galaxy_data, n_simulations=1000):
    """Run the temporal causality inversion test"""
    results = temporal_causality_inversion_test(cmb_data, galaxy_data, n_simulations)
    
    print("Creating visualization...")
    fig = visualize_results(results)
    
    output_file = "temporal_causality_results.png"
    print(f"Saving visualization to {output_file}...")
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Save detailed results to text file
    results_file = "temporal_causality_results.txt"
    print(f"Saving detailed results to {results_file}...")
    
    with open(results_file, 'w') as f:
        f.write("TEMPORAL CAUSALITY INVERSION TEST RESULTS\n")
        f.write("="*50 + "\n\n")
        
        f.write("Prediction of Galaxy Properties from CMB Patterns:\n")
        f.write(f"CMB Data R² Score: {results['actual_r2']:.4f}\n")
        f.write(f"Surrogate Data R² Score: {results['surrogate_mean']:.4f}\n")
        f.write(f"Ratio: {results['actual_r2']/results['surrogate_mean']:.2f}x\n")
        f.write(f"Z-score: {results['z_score']:.4f}\n")
        f.write(f"P-value: {results['p_value']:.8f}\n\n")
        
        f.write("Temporal Asymmetry Analysis:\n")
        f.write(f"Early-forming Galaxies R²: {results['early_r2']:.4f}\n")
        f.write(f"Late-forming Galaxies R²: {results['late_r2']:.4f}\n")
        f.write(f"Temporal Asymmetry: {results['temporal_asymmetry']:.4f}\n")
        f.write(f"Surrogate Asymmetry: {results['asymmetry_mean']:.4f}\n")
        f.write(f"Z-score: {results['asymmetry_z']:.4f}\n")
        f.write(f"P-value: {results['asymmetry_p']:.8f}\n\n")
        
        f.write("Phi-Optimization in Temporal Relationships:\n")
        f.write(f"Phi-related Formation Time R²: {results['phi_r2']:.4f}\n")
        f.write(f"Non-Phi-related Formation Time R²: {results['non_phi_r2']:.4f}\n")
        f.write(f"Phi Optimization: {results['phi_optimization']:.4f}\n")
        f.write(f"Surrogate Phi Optimization: {results['phi_opt_mean']:.4f}\n")
        f.write(f"Z-score: {results['phi_opt_z']:.4f}\n")
        f.write(f"P-value: {results['phi_opt_p']:.8f}\n\n")
        
        f.write(f"Test completed in {results['execution_time']:.2f} seconds\n")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Temporal Causality Inversion Test")
    parser.add_argument("--cmb", help="Path to CMB data file (FITS or HEALPix map)")
    parser.add_argument("--galaxies", help="Path to galaxy catalog file (CSV or FITS)")
    parser.add_argument("--simulations", type=int, default=1000, 
                        help="Number of Monte Carlo simulations (default: 1000)")
    parser.add_argument("--simulated", action="store_true",
                        help="Use simulated galaxy data (for testing only)")
    parser.add_argument("--scale55-enhanced", action="store_true",
                        help="Use scale 55 enhanced galaxy data with Golden Ratio bias")
    parser.add_argument("--phi-bias", type=float, default=0.3,
                        help="Strength of Golden Ratio bias in enhanced simulation (0-1)")
    
    args = parser.parse_args()
    
    # Load CMB data
    if args.cmb:
        cmb_data = load_cmb_data(args.cmb)
    else:
        print("No CMB data file specified. Please provide a path with --cmb")
        cmb_data = None
    
    # Load or simulate galaxy data
    if args.scale55_enhanced and cmb_data is not None:
        print("Creating scale 55 enhanced simulated galaxy catalog with Golden Ratio bias...")
        galaxy_data = create_scale55_enhanced_galaxy_data(cmb_data, phi_bias=args.phi_bias)
    elif args.simulated:
        galaxy_data = create_simulated_galaxy_data()
    elif args.galaxies:
        galaxy_data = load_galaxy_data(args.galaxies)
    else:
        print("No galaxy data file specified. Please provide a path with --galaxies")
        print("Or use --simulated for test data (not for scientific results)")
        galaxy_data = None
    
    # Run test if data is available
    if cmb_data is not None and galaxy_data is not None:
        main(cmb_data, galaxy_data, args.simulations)
    else:
        print("Cannot run test due to missing data. Please check error messages above.")

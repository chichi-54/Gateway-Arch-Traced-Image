import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.interpolate import splprep, splev
import matplotlib as mpl

def detect_pond_shape_smooth(image_path):
    """Detect the pond outline with smooth curves"""
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    print(f"Image loaded: {img.shape}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # Threshold to isolate dark pond area - adjust this value if needed
    _, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)
    
    # Clean up the mask with larger kernels for smoother edges
    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No contours found in the image")
    
    # Find the largest contour (the pond)
    pond_contour = max(contours, key=cv2.contourArea)
    
    # Use very gentle smoothing to preserve natural shape
    epsilon = 0.002 * cv2.arcLength(pond_contour, True)  # Reduced for more detail
    pond_contour = cv2.approxPolyDP(pond_contour, epsilon, True)
    
    pond_contour = pond_contour.squeeze()
    
    if pond_contour.ndim != 2 or pond_contour.shape[1] != 2:
        raise ValueError(f"Invalid contour shape: {pond_contour.shape}")
    
    return pond_contour

def create_smooth_contour(contour, smoothing_factor=0.01):
    """Apply spline interpolation for smooth curves"""
    # Close the contour
    x = np.append(contour[:, 0], contour[0, 0])
    y = np.append(contour[:, 1], contour[0, 1])
    
    # Fit a spline to the points
    tck, u = splprep([x, y], s=smoothing_factor * len(contour), per=1)
    
    # Evaluate the spline at more points for smoothness
    u_new = np.linspace(0, 1, max(500, len(contour) * 3))  # More points = smoother
    x_new, y_new = splev(u_new, tck)
    
    # Remove the duplicate closing point
    smoothed_contour = np.column_stack([x_new[:-1], y_new[:-1]])
    
    return smoothed_contour

def plot_scaled_pond_smooth(pond_contour, scale_bar_pixels, scale_bar_meters):
    """Plot the pond with beautiful smooth curves"""
    # Calculate scale
    meters_per_pixel = scale_bar_meters / scale_bar_pixels
    
    # Apply spline smoothing (FIXED: different variable name)
    smoothed_contour = create_smooth_contour(pond_contour, smoothing_factor=0.005)
    
    # Convert to meters
    pond_coords_m = smoothed_contour * meters_per_pixel
    
    # Calculate area (using original contour for accuracy)
    area_pixels = cv2.contourArea(pond_contour)
    area_m2 = area_pixels * (meters_per_pixel ** 2)
    
    # Create professional-looking plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot the pond with smooth curves and beautiful styling
    polygon = Polygon(pond_coords_m, closed=True, 
                     facecolor='#87CEEB',  # Base color
                     edgecolor='#1E3F66',   # Dark blue outline
                     linewidth=3.5,
                     alpha=0.9,
                     linestyle='-',
                     joinstyle='round',  # Smooth line joins
                     capstyle='round')   # Smooth line caps
    ax.add_patch(polygon)
    
    # Add a subtle inner glow effect
    polygon_inner = Polygon(pond_coords_m, closed=True, 
                           fill=False,
                           edgecolor='white',
                           linewidth=1.5,
                           alpha=0.3,
                           linestyle='-')
    ax.add_patch(polygon_inner)
    
    # Set equal aspect ratio for accurate scaling
    ax.set_aspect('equal')
    
    # Calculate plot limits with elegant margins
    x_coords = pond_coords_m[:, 0]
    y_coords = pond_coords_m[:, 1]
    
    x_range = x_coords.max() - x_coords.min()
    y_range = y_coords.max() - y_coords.min()
    
    margin_x = x_range * 0.12
    margin_y = y_range * 0.12
    
    x_min = x_coords.min() - margin_x
    x_max = x_coords.max() + margin_x
    y_min = y_coords.min() - margin_y
    y_max = y_coords.max() + margin_y
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Professional styling
    ax.set_xlabel('East-West Distance (meters)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('North-South Distance (meters)', fontsize=13, fontweight='bold', labelpad=10)
    
    # Competition-quality title
    ax.set_title('Gateway Arch Reflection Pond\nHydraulic Analysis Model', 
                fontsize=18, fontweight='bold', pad=20,
                color='#2C3E50')
    
    # Enhanced grid
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, color='gray')
    ax.grid(True, which='minor', alpha=0.1, linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    
    # Professional scale bar
    scale_length = 50  # 50 meter scale bar
    scale_y_position = y_min + margin_y * 0.3
    
    ax.plot([x_min + margin_x * 0.5, x_min + margin_x * 0.5 + scale_length], 
            [scale_y_position, scale_y_position], 
            color='#2C3E50', linewidth=5, solid_capstyle='round')
    
    ax.text(x_min + margin_x * 0.5 + scale_length/2, scale_y_position + margin_y * 0.08,
            f'{scale_length} METERS', ha='center', fontsize=11, fontweight='bold',
            color='#2C3E50',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='white', edgecolor='#2C3E50', alpha=0.9))
    
    # Elegant information box
    info_text = f'Surface Area: {area_m2:,.0f} m²\n' \
                f'Scale: 1:{(1/meters_per_pixel):.0f}'
    
    ax.text(0.98, 0.98, info_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.6", facecolor='white', 
                     edgecolor='#2C3E50', alpha=0.95),
            fontweight='bold', color='#2C3E50')
    
    # Add dimension labels
    width_m = x_coords.max() - x_coords.min()
    height_m = y_coords.max() - y_coords.min()
    
    dim_text = f'Dimensions: {width_m:.1f} × {height_m:.1f} m'
    ax.text(0.02, 0.02, dim_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.9))
    
    # Style the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_color('#7F8C8D')
    ax.spines['bottom'].set_color('#7F8C8D')
    
    # Professional tick formatting
    ax.tick_params(axis='both', which='major', labelsize=10, colors='#2C3E50')
    
    # Add a subtle background gradient
    ax.set_facecolor('#F8F9F9')
    
    plt.tight_layout()
    
    print(f"✓ Smooth pond outline generated with {len(pond_coords_m)} points")
    print(f"✓ Surface Area: {area_m2:,.0f} square meters")
    print(f"✓ Dimensions: {width_m:.1f} × {height_m:.1f} meters")
    print(f"✓ Scale: 1 pixel = {meters_per_pixel:.4f} meters")
    
    return fig, ax, pond_coords_m

# MAIN CODE - COMPETITION QUALITY
if __name__ == "__main__":
    # Your image path
    img_path = r"C:\Users\cheta\Downloads\Gateway Pond photos\gatewaypond.jpg"
    
    # SCALE CALIBRATION - Update these based on your image
    scale_bar_pixels = 200   # Measure scale bar in pixels
    scale_bar_meters = 100   # Real distance in meters
    
    try:
        print("🔄 Detecting pond outline with smooth curves...")
        pond_contour = detect_pond_shape_smooth(img_path)
        print(f"✓ Raw contour points: {len(pond_contour)}")
        
        print("🎨 Creating competition-quality plot...")
        fig, ax, pond_coords_m = plot_scaled_pond_smooth(pond_contour, scale_bar_pixels, scale_bar_meters)
        
        # Save high-resolution version for competition
        plt.savefig('Gateway_Arch_Pond_Competition.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print("💾 High-resolution plot saved as 'Gateway_Arch_Pond_Competition.png'")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


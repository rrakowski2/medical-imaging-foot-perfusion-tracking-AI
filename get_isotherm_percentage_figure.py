'''
Function to generate the isotherm percentage figure - for the foot-tracking app
'''


# Set compute invironment
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def get_percentage_figure(x_axis, white_ratios, yellow_ratios, orange_ratios, red_ratios, blue_ratios, rounded_max_value):
    # Create a percentage figure
    fig2, ax2 = plt.subplots()
    ax2.plot(x_axis, white_ratios, label=f'>{rounded_max_value+0.5} [°C]', color='white')
    ax2.plot(x_axis, yellow_ratios, label=f'{rounded_max_value-0.5}-{rounded_max_value+0.5}', color='yellow')
    ax2.plot(x_axis, orange_ratios, label=f'{rounded_max_value-1.5}-{rounded_max_value-0.5}', color='orange')
    ax2.plot(x_axis, red_ratios, label=f'{rounded_max_value-2.5}-{rounded_max_value-1.5}', color='red')
    ax2.plot(x_axis, blue_ratios, label=f'<{rounded_max_value-2.5}', color='blue')
    

    # Set font size for axes titles and add numerical labels
    ax2.set_xlabel('Frame no.', fontsize=14, color='white')
    ax2.set_ylabel('Temp range area ratios [%]', fontsize=14, color='white')


    # Enforce integer ticks on both x-axis and y-axis for clarity
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))


    # Change background color to black
    fig2.patch.set_facecolor('black')
    ax2.set_facecolor('black')


    # Adjust legend position (keeping it intact)
    legend = ax2.legend(loc='upper left', bbox_to_anchor=(1, 0.86), facecolor='black', edgecolor='white', fontsize=10, labelcolor='white')
    
    
    # Add numerical labels to axes
    for label in ax2.get_xticklabels():
        label.set_color('white')
    for label in ax2.get_yticklabels():
        label.set_color('white')


    # Define text placement variables
    x_offset = 1.02  # Align with the legend
    y_offset_start = 0.89  # Move "Temp bands" closer to the legend
    y_step = -0.05  # Space between lines
    # Add "Temperature Bands" title above the legend
    ax2.text(x_offset, y_offset_start, "Temp bands", color='white', fontsize=12, fontweight='bold', transform=ax2.transAxes)
    
    
    # Move the "Present / Reference" text below the legend
    y_offset_start = 0.39  # Lower position for ratio values
    ax2.text(x_offset, y_offset_start, "Now / Ref ratio", color='lightgray', fontsize=12, fontweight='bold', transform=ax2.transAxes)
    
    
    # Remove color names and keep numerical values only
    ax2.text(x_offset, y_offset_start + y_step * 1.5, f"{white_ratios[-1]:.1f}% / {white_ratios[0]:.1f}%", color='white', fontsize=13, transform=ax2.transAxes)
    ax2.text(x_offset, y_offset_start + y_step * 2.5, f"{yellow_ratios[-1]:.1f}% / {yellow_ratios[0]:.1f}%", color='yellow', fontsize=13, transform=ax2.transAxes)
    ax2.text(x_offset, y_offset_start + y_step * 3.5, f"{orange_ratios[-1]:.1f}% / {orange_ratios[0]:.1f}%", color='orange', fontsize=13, transform=ax2.transAxes)
    ax2.text(x_offset, y_offset_start + y_step * 4.5, f"{red_ratios[-1]:.1f}% / {red_ratios[0]:.1f}%", color='red', fontsize=13, transform=ax2.transAxes)
    ax2.text(x_offset, y_offset_start + y_step * 5.5, f"{blue_ratios[-1]:.1f}% / {blue_ratios[0]:.1f}%", color='blue', fontsize=13, transform=ax2.transAxes)
    
    
    # Expand layout to ensure proper spacing
    plt.tight_layout(rect=[0, 0, 0.99, 0.95])
    plt.title(f"Isotherm percentage-right (Tmax={rounded_max_value}°C)",  color='white', fontsize=18, loc='center')
    

    # Save the figure
    temp_file2 = 'temporary_percent_evolution.png'
    fig2.savefig(temp_file2, facecolor='black')
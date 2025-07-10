#!/usr/bin/python

'''
Function to plot temperature evolution figure - for the foot-tracking app
by Rafal June 2025
''' 


# Set compute environment
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import get_bounding_box  # gives access to x_axis and y


labels = ['LP', 'MP', 'LC', 'MC', 'GT']
colors = ['red', 'blue', 'green', 'purple', 'orange']


def plot_series():
    if not get_bounding_box.x_axis or not any(get_bounding_box.y):
        print("No data to plot yet.")
        return


    fig, ax = plt.subplots()
    for idx in range(5):
        ax.plot(get_bounding_box.x_axis, get_bounding_box.y[idx], label=labels[idx], color=colors[idx])
    

    # Set font size for axes titles
    ax.set_xlabel('Frame no.', fontsize=14)
    ax.set_ylabel('Temp-max [°C]', fontsize=14)


    # Enforce integer ticks on both x-axis and y-axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    

    # Position the legend outside the plot axes
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))


    # Automatically adjust the layout
    plt.tight_layout(rect=[0, 0, 0.99, 0.95])                                                                     
    plt.ticklabel_format(style='plain', axis='x')
    plt.title("Angiosomes' max temp - right", fontsize=18)
    #plt.show(block=False)  # Ensure the figures don't block execution


    # Save the figure to a temporary file
    temp_file = 'temporary_temp_evolution.png'
    fig.savefig(temp_file)                                                    
    plt.close('all')     
  

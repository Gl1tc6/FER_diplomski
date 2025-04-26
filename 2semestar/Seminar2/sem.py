import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patheffects as path_effects

# Define the data
activities = [
    {"start": None, "end": "2025-03-07", "name": "Dogovor teme seminara s mentorom", "color": "#eeeeee"},
    {"start": "2025-03-07", "end": "2025-04-01", "name": "Predaja nacrta seminarskog rada", "color": "white"},
    {"start": "2025-04-01", "end": "2025-04-08", "name": "Ispravak nacrta", "color": "#eeeeee"},
    {"start": "2025-04-08", "end": "2025-04-25", "name": "Predaja 1. verzije seminarskog rada", "color": "white"},
    {"start": "2025-04-25", "end": "2025-05-21", "name": "Formiranje i objava prezentacijskih grupa", "color": "#eeeeee"},
    {"start": "2025-05-21", "end": "2025-05-22", "name": "Predaja konačne verzije seminarskog rada i prezentacije", "color": "white"},
    {"start": "2025-05-22", "end": "2025-05-23", "name": "Određivanje termina prezentacije", "color": "#eeeeee"},
    {"start": "2025-05-26", "end": "2025-06-14", "name": "Prezentacija rada", "color": "white"},
    {"start": "2025-06-14", "end": "2025-06-30", "name": "Zadnji dan raspodjele bodova studenata", "color": "#eeeeee"}
]

# Format dates
def parse_date(date_str):
    if date_str is None:
        # Use February as the beginning for visualization purposes
        return datetime.strptime("2025-02-15", "%Y-%m-%d")
    return datetime.strptime(date_str, "%Y-%m-%d")

# Create figure and axis
plt.figure(figsize=(14, 8))
ax = plt.subplot(111)

# Y-axis positions for each activity
positions = np.arange(len(activities))

# Plot activities as horizontal bars
for i, activity in enumerate(activities):
    start_date = parse_date(activity["start"])
    end_date = parse_date(activity["end"])
    
    # Calculate duration in days for annotation positioning
    duration = (end_date - start_date).days
    
    # Draw bar with background color
    rect = Rectangle((mdates.date2num(start_date), i-0.4), 
                    mdates.date2num(end_date) - mdates.date2num(start_date), 
                    0.8, 
                    color=activity["color"],
                    ec='black',
                    linewidth=1.2)
    ax.add_patch(rect)
    
    # Add activity name text
    text = ax.text(
        mdates.date2num(start_date) + (mdates.date2num(end_date) - mdates.date2num(start_date))/2, 
        i, 
        activity["name"],
        ha='center',
        va='center',
        fontweight='bold',
        fontsize=10,
        wrap=True
    )
    
    # Add outline to text for better readability
    text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

# Format x-axis as dates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=45)

# Set y-axis
plt.yticks([])  # Hide y-axis ticks
plt.ylim(-0.5, len(activities) - 0.5)

# Add key dates as vertical lines and labels
key_dates = [
    {"date": "2025-03-07", "label": "07.03 - Rok za dogovor teme"},
    {"date": "2025-04-01", "label": "01.04 - Rok za predaju nacrta"},
    {"date": "2025-04-08", "label": "08.04 - Rok za ispravak nacrta"},
    {"date": "2025-04-25", "label": "25.04 - Rok za 1. verziju"},
    {"date": "2025-05-21", "label": "21.05 - Rok za objavu grupa"},
    {"date": "2025-05-22", "label": "22.05 - Rok za konačnu verziju"},
    {"date": "2025-06-14", "label": "14.06 - Kraj prezentacija"},
    {"date": "2025-06-30", "label": "30.06 - Zadnji dan raspodjele bodova"}
]

for date_info in key_dates:
    date = datetime.strptime(date_info["date"], "%Y-%m-%d")
    ax.axvline(x=date, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    # Add date label at the top
    ax.text(
        mdates.date2num(date), 
        len(activities) - 0.3,
        date_info["label"],
        rotation=45,
        ha='right',
        va='bottom',
        fontsize=9,
        color='red',
        fontweight='bold'
    )

# Add title and grid
plt.title('Raspored aktivnosti za predmet Seminar 2024/2025', fontsize=16, pad=20)
plt.grid(axis='x', linestyle='--', alpha=0.5)

# Adjust layout for better display
plt.tight_layout()

# Add subtitle with additional info
plt.figtext(0.5, 0.01, 
           "Fakultet elektrotehnike i računarstva - Seminar - Plan aktivnosti",
           ha='center', fontsize=10, style='italic')

# Save the figure
plt.savefig('seminar_timeline.png', dpi=300, bbox_inches='tight')
print("Image saved as 'seminar_timeline.png'")

# Show the plot (optional, comment out if running in environment without display)
plt.show()

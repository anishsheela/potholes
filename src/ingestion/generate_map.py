import pandas as pd
import folium
import argparse
import os

def generate_map(csv_path, output_map_path):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Convert lat/lon strings to floats
    # Handling potential trailing characters or extra dots gracefully
    def safe_float(val):
        try:
            # Assuming format like '8.5853' or '76.9872'
            # Remove any non-numeric/dot characters just in case OCR grabbed noise
            clean_val = ''.join(c for c in str(val) if c.isdigit() or c == '.')
            
            # If there are multiple dots (OCR error), keep only the first one
            parts = clean_val.split('.')
            if len(parts) > 2:
                clean_val = parts[0] + '.' + ''.join(parts[1:])
                
            return float(clean_val)
        except ValueError:
            return None

    df['Lat_Float'] = df['Latitude'].apply(safe_float)
    df['Lon_Float'] = df['Longitude'].apply(safe_float)

    # Drop rows with invalid coordinates
    initial_len = len(df)
    df = df.dropna(subset=['Lat_Float', 'Lon_Float'])
    print(f"Dropped {initial_len - len(df)} rows with unparseable coordinates.")

    if df.empty:
        print("Error: No valid GPS data to plot.")
        return

    # Extract coordinates as a list of tuples
    locations = list(zip(df['Lat_Float'], df['Lon_Float']))
    
    # Calculate map center (average of all points)
    center_lat = sum(loc[0] for loc in locations) / len(locations)
    center_lon = sum(loc[1] for loc in locations) / len(locations)

    print(f"Generating map centered at {center_lat:.4f}, {center_lon:.4f} with {len(locations)} total waypoints...")

    # Create base map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="OpenStreetMap")

    # Group data by Video to create distinct segments
    # This prevents Folium from drawing a massive straight line from the end of Video A to the start of Video B
    grouped = df.groupby('Video')
    
    # Pre-defined list of colors to cycle through for each segment
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'darkred', 'cadetblue', 'darkgreen', 'darkblue', 'darkpurple']
    
    for i, (video_name, group) in enumerate(grouped):
        color = colors[i % len(colors)]
        
        segment_locations = list(zip(group['Lat_Float'], group['Lon_Float']))
        
        if len(segment_locations) < 2:
            continue
            
        # Add the route line segment
        folium.PolyLine(
            segment_locations,
            weight=5,
            color=color,
            opacity=0.8,
            tooltip=f"{video_name}"
        ).add_to(m)

        # Add start marker for this segment
        folium.Marker(
            location=segment_locations[0],
            popup=f"<b>{video_name} (Start)</b><br>Time: {group.iloc[0]['Date']} {group.iloc[0]['Time']}<br>Speed: {group.iloc[0]['Speed_kmh']} km/h",
            icon=folium.Icon(color=color, icon="play")
        ).add_to(m)

        # Add end marker for this segment
        folium.Marker(
            location=segment_locations[-1],
            popup=f"<b>{video_name} (End)</b><br>Time: {group.iloc[-1]['Date']} {group.iloc[-1]['Time']}<br>Speed: {group.iloc[-1]['Speed_kmh']} km/h",
            icon=folium.Icon(color=color, icon="stop")
        ).add_to(m)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_map_path), exist_ok=True)
    
    # Save the map
    m.save(output_map_path)
    print(f"Successfully saved map to {output_map_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default="output/route_data.csv", help="Input CSV file path")
    parser.add_argument("--output", "-o", type=str, default="output/map.html", help="Output HTML map path")
    args = parser.parse_args()
    
    generate_map(args.input, args.output)

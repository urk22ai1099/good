"""
AgroMind+ Data Generator
Generates realistic synthetic agricultural data with temporal patterns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


class AgriculturalDataGenerator:
    def __init__(self, n_records=5000, weeks=52):
        """
        Initialize data generator
        """
        self.n_records = n_records
        self.weeks = weeks

        # FIX 1: Fixed farm IDs (instead of random every row)
        self.farm_ids = [f"FARM_{i}" for i in range(1, 201)]  # 200 farms

        # Crop definitions
        self.crops = {
            'Aman_Rice': {
                'N': (120, 180), 'P': (40, 55), 'K': (40, 55),
                'pH': (6.0, 7.5), 'temp': (25, 32), 'humidity': (70, 90),
                'moisture': (70, 85), 'rainfall': (150, 200), 'sunlight': (5, 7)
            },
            'Boro_Rice': {
                'N': (150, 200), 'P': (45, 60), 'K': (45, 60),
                'pH': (6.5, 7.5), 'temp': (22, 30), 'humidity': (65, 85),
                'moisture': (75, 90), 'rainfall': (80, 120), 'sunlight': (6, 8)
            },
            'Wheat': {
                'N': (100, 150), 'P': (35, 50), 'K': (35, 50),
                'pH': (6.0, 7.5), 'temp': (18, 25), 'humidity': (50, 70),
                'moisture': (50, 70), 'rainfall': (50, 80), 'sunlight': (7, 9)
            },
            'Maize': {
                'N': (120, 160), 'P': (40, 55), 'K': (40, 55),
                'pH': (5.8, 7.0), 'temp': (21, 30), 'humidity': (60, 80),
                'moisture': (55, 75), 'rainfall': (60, 100), 'sunlight': (6, 8)
            },
            'Millets': {
                'N': (80, 120), 'P': (25, 40), 'K': (30, 45),
                'pH': (6.0, 8.0), 'temp': (25, 35), 'humidity': (40, 65),
                'moisture': (40, 60), 'rainfall': (30, 60), 'sunlight': (7, 10)
            },
            'Pulses': {
                'N': (40, 80), 'P': (30, 45), 'K': (35, 50),
                'pH': (6.0, 7.5), 'temp': (20, 30), 'humidity': (55, 75),
                'moisture': (50, 70), 'rainfall': (50, 90), 'sunlight': (6, 8)
            },
            'Cotton': {
                'N': (100, 140), 'P': (35, 50), 'K': (40, 60),
                'pH': (6.5, 8.0), 'temp': (25, 35), 'humidity': (50, 70),
                'moisture': (50, 70), 'rainfall': (60, 100), 'sunlight': (7, 9)
            }
        }

        self.irrigation_types = ['Drip', 'Canal', 'Rainfed', 'Sprinkler']
        self.regions = ['North', 'South', 'East', 'West', 'Central']

    def generate_temporal_pattern(self, crop_name, week):
        crop_params = self.crops[crop_name]

        season_factor = np.sin(2 * np.pi * week / 52)

        temp_mean = np.mean(crop_params['temp'])
        temp_std = (crop_params['temp'][1] - crop_params['temp'][0]) / 4
        temperature = temp_mean + season_factor * temp_std + np.random.normal(0, 2)
        temperature = np.clip(temperature, 15, 40)

        if 20 <= week <= 35:
            rainfall = np.random.uniform(
                crop_params['rainfall'][1] * 0.8,
                crop_params['rainfall'][1] * 1.2
            )
        else:
            rainfall = np.random.uniform(
                crop_params['rainfall'][0] * 0.5,
                crop_params['rainfall'][0] * 1.5
            )

        humidity = np.mean(crop_params['humidity']) + (rainfall / 200) * 20
        humidity = np.clip(humidity + np.random.normal(0, 5), 30, 95)

        return {
            'N': np.random.uniform(*crop_params['N']),
            'P': np.random.uniform(*crop_params['P']),
            'K': np.random.uniform(*crop_params['K']),
            'pH': np.random.uniform(*crop_params['pH']),
            'Temperature': temperature,
            'Humidity': humidity,
            'Moisture': np.random.uniform(*crop_params['moisture']),
            'Rainfall': rainfall,
            'Sunlight': np.random.uniform(*crop_params['sunlight'])
        }

    def generate_dataset(self):
        print("Generating agricultural dataset...")

        records = []
        start_date = datetime.now() - timedelta(weeks=self.weeks)

        # FIX 2: Temporal continuity per farm
        for farm in self.farm_ids:
            crop = random.choice(list(self.crops.keys()))
            region = random.choice(self.regions)
            irrigation = random.choice(self.irrigation_types)

            for week in range(self.weeks):
                timestamp = start_date + timedelta(weeks=week)
                data = self.generate_temporal_pattern(crop, week)

                record = {
                    'Timestamp': timestamp,
                    'Week': week,
                    'Region': region,
                    'FarmID': farm,
                    **data,
                    'PreviousCrop': crop,
                    'IrrigationType': irrigation,
                    'Crop': crop,
                    'Yield': self._calculate_yield(crop, data)
                }

                records.append(record)

        df = pd.DataFrame(records).sort_values('Timestamp').reset_index(drop=True)
        print(f"✓ Dataset generated: {df.shape}")
        return df

    def _calculate_yield(self, crop, data):
        base_yield = {
            'Aman_Rice': 4.5, 'Boro_Rice': 5.5, 'Wheat': 4.0,
            'Maize': 6.0, 'Millets': 2.5, 'Pulses': 2.0, 'Cotton': 3.5
        }
        return max(0.5, base_yield[crop] + np.random.normal(0, 0.3))

    def create_sequences(self, df, sequence_length=4):
        # FIX 3: Correct list initialization
        sequences = []
        labels = []

        feature_cols = [
            'N', 'P', 'K', 'pH',
            'Temperature', 'Humidity',
            'Moisture', 'Rainfall', 'Sunlight'
        ]

        for farm in df['FarmID'].unique():
            farm_data = df[df['FarmID'] == farm].sort_values('Timestamp')

            if len(farm_data) <= sequence_length:
                continue

            for i in range(len(farm_data) - sequence_length):
                seq = farm_data.iloc[i:i + sequence_length][feature_cols].values
                target = farm_data.iloc[i + sequence_length]['Crop']

                sequences.append(seq)
                labels.append(target)

        print(f"✓ Created {len(sequences)} sequences")
        return np.array(sequences), np.array(labels)


def main():
    print("=" * 80)
    print("AgroMind+ Temporal Agricultural Data Generator")
    print("=" * 80)

    generator = AgriculturalDataGenerator()
    df = generator.generate_dataset()

    df.to_csv("../data/agricultural_data.csv", index=False)
    print("✓ Dataset saved")

    sequences, labels = generator.create_sequences(df, sequence_length=4)

    np.save("../data/sequences.npy", sequences)
    np.save("../data/labels.npy", labels)

    print(f"Sequences shape: {sequences.shape}")
    print(f"Labels shape: {labels.shape}")


if __name__ == "__main__":
    main()

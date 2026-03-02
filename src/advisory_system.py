"""
AgroMind+ Adaptive Crop Advisory Intelligence (ACAI)
Provides fertilizer, irrigation, and yield recommendations
"""

import numpy as np
import pandas as pd
from datetime import datetime
import joblib

class AdaptiveCropAdvisory:
    def __init__(self):
        """Initialize advisory system"""
        self.fertilizer_db = self._init_fertilizer_database()
        self.irrigation_db = self._init_irrigation_database()
        self.psi_weights = {
            'water_efficiency': 0.3,
            'fertilizer_efficiency': 0.3,
            'yield_potential': 0.25,
            'sustainability': 0.15
        }
    
    def _init_fertilizer_database(self):
        """Initialize fertilizer recommendations database"""
        return {
            'Aman_Rice': {
                'base': {'Urea': 150, 'DAP': 100, 'MOP': 80},
                'N_deficit': 'Urea', 'P_deficit': 'DAP', 'K_deficit': 'MOP',
                'optimal_N': 150, 'optimal_P': 50, 'optimal_K': 50
            },
            'Boro_Rice': {
                'base': {'Urea': 180, 'DAP': 120, 'MOP': 90},
                'N_deficit': 'Urea', 'P_deficit': 'DAP', 'K_deficit': 'MOP',
                'optimal_N': 175, 'optimal_P': 55, 'optimal_K': 55
            },
            'Wheat': {
                'base': {'Urea': 120, 'DAP': 80, 'MOP': 60},
                'N_deficit': 'Urea', 'P_deficit': 'DAP', 'K_deficit': 'MOP',
                'optimal_N': 125, 'optimal_P': 45, 'optimal_K': 45
            },
            'Maize': {
                'base': {'Urea': 140, 'DAP': 90, 'MOP': 70},
                'N_deficit': 'Urea', 'P_deficit': 'DAP', 'K_deficit': 'MOP',
                'optimal_N': 140, 'optimal_P': 48, 'optimal_K': 48
            },
            'Millets': {
                'base': {'Urea': 80, 'DAP': 60, 'MOP': 50},
                'N_deficit': 'Urea', 'P_deficit': 'DAP', 'K_deficit': 'MOP',
                'optimal_N': 100, 'optimal_P': 35, 'optimal_K': 38
            },
            'Pulses': {
                'base': {'Urea': 40, 'DAP': 70, 'MOP': 55},
                'N_deficit': 'Urea', 'P_deficit': 'DAP', 'K_deficit': 'MOP',
                'optimal_N': 60, 'optimal_P': 40, 'optimal_K': 43
            },
            'Cotton': {
                'base': {'Urea': 120, 'DAP': 85, 'MOP': 75},
                'N_deficit': 'Urea', 'P_deficit': 'DAP', 'K_deficit': 'MOP',
                'optimal_N': 120, 'optimal_P': 43, 'optimal_K': 50
            }
        }
    
    def _init_irrigation_database(self):
        """Initialize irrigation recommendations"""
        return {
            'Aman_Rice': {
                'type': 'Canal',
                'frequency': 'Every 3-4 days',
                'water_depth': '5-7 cm',
                'critical_stages': ['Tillering', 'Flowering', 'Grain filling'],
                'total_water_mm': 1200
            },
            'Boro_Rice': {
                'type': 'Canal',
                'frequency': 'Every 2-3 days',
                'water_depth': '5-8 cm',
                'critical_stages': ['Transplanting', 'Tillering', 'Panicle initiation'],
                'total_water_mm': 1400
            },
            'Wheat': {
                'type': 'Sprinkler',
                'frequency': 'Every 7-10 days',
                'water_depth': '4-6 cm',
                'critical_stages': ['Crown root', 'Jointing', 'Flowering'],
                'total_water_mm': 400
            },
            'Maize': {
                'type': 'Drip',
                'frequency': 'Every 5-7 days',
                'water_depth': '5-7 cm',
                'critical_stages': ['Knee-high', 'Tasseling', 'Silking'],
                'total_water_mm': 600
            },
            'Millets': {
                'type': 'Rainfed',
                'frequency': 'As needed (1-2 times)',
                'water_depth': '3-5 cm',
                'critical_stages': ['Flowering', 'Grain filling'],
                'total_water_mm': 350
            },
            'Pulses': {
                'type': 'Sprinkler',
                'frequency': 'Every 10-12 days',
                'water_depth': '4-6 cm',
                'critical_stages': ['Branching', 'Flowering', 'Pod formation'],
                'total_water_mm': 450
            },
            'Cotton': {
                'type': 'Drip',
                'frequency': 'Every 5-7 days',
                'water_depth': '5-7 cm',
                'critical_stages': ['Squaring', 'Flowering', 'Boll development'],
                'total_water_mm': 700
            }
        }
    
    def analyze_soil_conditions(self, soil_data, crop):
        """Analyze current soil conditions against crop requirements"""
        N_current = soil_data.get('N', 0)
        P_current = soil_data.get('P', 0)
        K_current = soil_data.get('K', 0)
        pH_current = soil_data.get('pH', 7.0)
        
        crop_fert = self.fertilizer_db.get(crop, self.fertilizer_db['Wheat'])
        
        # Calculate deficits
        N_deficit = max(0, crop_fert['optimal_N'] - N_current)
        P_deficit = max(0, crop_fert['optimal_P'] - P_current)
        K_deficit = max(0, crop_fert['optimal_K'] - K_current)
        
        return {
            'N_deficit': N_deficit,
            'P_deficit': P_deficit,
            'K_deficit': K_deficit,
            'pH_current': pH_current,
            'needs_amendment': N_deficit > 20 or P_deficit > 10 or K_deficit > 10
        }
    
    def generate_fertilizer_plan(self, soil_analysis, crop, farm_size_ha=1.0):
        """Generate detailed fertilizer application plan"""
        crop_fert = self.fertilizer_db.get(crop, self.fertilizer_db['Wheat'])
        
        fertilizer_plan = {
            'crop': crop,
            'farm_size_ha': farm_size_ha,
            'fertilizers': []
        }
        
        # Nitrogen
        if soil_analysis['N_deficit'] > 10:
            urea_kg = (crop_fert['base']['Urea'] + soil_analysis['N_deficit'] * 0.5) * farm_size_ha
            fertilizer_plan['fertilizers'].append({
                'name': 'Urea (46% N)',
                'quantity_kg': round(urea_kg, 1),
                'application': 'Split application: 50% at sowing, 30% at tillering, 20% at flowering',
                'deficit_addressed': f"Nitrogen deficit: {soil_analysis['N_deficit']:.1f} kg/ha"
            })
        
        # Phosphorus
        if soil_analysis['P_deficit'] > 5:
            dap_kg = (crop_fert['base']['DAP'] + soil_analysis['P_deficit'] * 0.8) * farm_size_ha
            fertilizer_plan['fertilizers'].append({
                'name': 'DAP (18% N, 46% P₂O₅)',
                'quantity_kg': round(dap_kg, 1),
                'application': 'Apply at sowing/transplanting',
                'deficit_addressed': f"Phosphorus deficit: {soil_analysis['P_deficit']:.1f} kg/ha"
            })
        
        # Potassium
        if soil_analysis['K_deficit'] > 5:
            mop_kg = (crop_fert['base']['MOP'] + soil_analysis['K_deficit'] * 0.7) * farm_size_ha
            fertilizer_plan['fertilizers'].append({
                'name': 'MOP (60% K₂O)',
                'quantity_kg': round(mop_kg, 1),
                'application': 'Split application: 60% at sowing, 40% at flowering',
                'deficit_addressed': f"Potassium deficit: {soil_analysis['K_deficit']:.1f} kg/ha"
            })
        
        # pH correction
        if soil_analysis['pH_current'] < 6.0:
            lime_kg = (6.5 - soil_analysis['pH_current']) * 500 * farm_size_ha
            fertilizer_plan['fertilizers'].append({
                'name': 'Agricultural Lime',
                'quantity_kg': round(lime_kg, 1),
                'application': 'Apply 2-3 weeks before sowing',
                'deficit_addressed': f"pH too low: {soil_analysis['pH_current']:.1f}"
            })
        elif soil_analysis['pH_current'] > 8.0:
            gypsum_kg = (soil_analysis['pH_current'] - 7.5) * 400 * farm_size_ha
            fertilizer_plan['fertilizers'].append({
                'name': 'Gypsum',
                'quantity_kg': round(gypsum_kg, 1),
                'application': 'Apply and mix well before sowing',
                'deficit_addressed': f"pH too high: {soil_analysis['pH_current']:.1f}"
            })
        
        return fertilizer_plan
    
    def generate_irrigation_plan(self, crop, climate_data):
        """Generate irrigation schedule"""
        irrigation_info = self.irrigation_db.get(crop, self.irrigation_db['Wheat'])
        
        rainfall = climate_data.get('Rainfall', 50)
        temperature = climate_data.get('Temperature', 25)
        humidity = climate_data.get('Humidity', 70)
        
        # Adjust irrigation based on rainfall
        if rainfall > 100:
            frequency_multiplier = 0.7
        elif rainfall < 30:
            frequency_multiplier = 1.3
        else:
            frequency_multiplier = 1.0
        
        plan = {
            'crop': crop,
            'recommended_type': irrigation_info['type'],
            'frequency': irrigation_info['frequency'],
            'water_depth': irrigation_info['water_depth'],
            'critical_stages': irrigation_info['critical_stages'],
            'total_water_required_mm': irrigation_info['total_water_mm'],
            'adjustments': []
        }
        
        # Environmental adjustments
        if temperature > 32:
            plan['adjustments'].append({
                'factor': 'High temperature',
                'action': 'Increase frequency by 20%',
                'reason': f'Temperature {temperature}°C is high'
            })
        
        if humidity < 50:
            plan['adjustments'].append({
                'factor': 'Low humidity',
                'action': 'Monitor soil moisture closely',
                'reason': f'Humidity {humidity}% increases evaporation'
            })
        
        if rainfall > 100:
            plan['adjustments'].append({
                'factor': 'High rainfall',
                'action': 'Reduce irrigation by 30%',
                'reason': f'Rainfall {rainfall}mm is adequate'
            })
        
        return plan
    
    def predict_yield(self, crop, soil_data, climate_data, fertilizer_applied=True):
        """Predict expected yield"""
        base_yields = {
            'Aman_Rice': 4.5, 'Boro_Rice': 5.5, 'Wheat': 4.0,
            'Maize': 6.0, 'Millets': 2.5, 'Pulses': 2.0, 'Cotton': 3.5
        }
        
        base_yield = base_yields.get(crop, 3.0)
        
        # Soil factors
        N_factor = min(1.0, soil_data.get('N', 100) / 150)
        P_factor = min(1.0, soil_data.get('P', 30) / 50)
        K_factor = min(1.0, soil_data.get('K', 30) / 50)
        
        # Climate factors
        temp_optimal = {'Aman_Rice': 28, 'Boro_Rice': 26, 'Wheat': 22, 
                        'Maize': 25, 'Millets': 30, 'Pulses': 25, 'Cotton': 30}
        temp_factor = 1 - abs(climate_data.get('Temperature', 25) - temp_optimal.get(crop, 25)) / 20
        temp_factor = max(0.5, min(1.0, temp_factor))
        
        rainfall_factor = min(1.0, climate_data.get('Rainfall', 100) / 100)
        
        # Calculate yield
        yield_multiplier = (N_factor + P_factor + K_factor + temp_factor + rainfall_factor) / 5
        
        if fertilizer_applied:
            yield_multiplier *= 1.15  # 15% boost with proper fertilization
        
        predicted_yield = base_yield * yield_multiplier
        
        return {
            'crop': crop,
            'predicted_yield_t_ha': round(predicted_yield, 2),
            'confidence': 'High' if yield_multiplier > 0.8 else 'Medium',
            'factors': {
                'soil_nutrition': round((N_factor + P_factor + K_factor) / 3, 2),
                'climate_suitability': round((temp_factor + rainfall_factor) / 2, 2),
                'management': 1.15 if fertilizer_applied else 1.0
            }
        }
    
    def calculate_psi(self, crop, soil_data, climate_data):
        """Calculate Predictive Sustainability Index"""
        irrigation_info = self.irrigation_db[crop]
        
        # Water efficiency score (0-1)
        water_use = irrigation_info['total_water_mm']
        water_efficiency = 1 - (water_use / 1500)  # Normalized
        
        # Fertilizer efficiency score
        fert_info = self.fertilizer_db[crop]
        total_fert = sum(fert_info['base'].values())
        fertilizer_efficiency = 1 - (total_fert / 400)  # Normalized
        
        # Yield potential (from prediction)
        yield_data = self.predict_yield(crop, soil_data, climate_data)
        yield_potential = yield_data['predicted_yield_t_ha'] / 8.0  # Normalized
        
        # Sustainability (inverse of environmental impact)
        sustainability = (water_efficiency + fertilizer_efficiency) / 2
        
        # Calculate weighted PSI
        psi = (
            self.psi_weights['water_efficiency'] * water_efficiency +
            self.psi_weights['fertilizer_efficiency'] * fertilizer_efficiency +
            self.psi_weights['yield_potential'] * yield_potential +
            self.psi_weights['sustainability'] * sustainability
        )
        
        return {
            'psi_score': round(psi, 3),
            'psi_percentage': round(psi * 100, 1),
            'rating': 'Excellent' if psi > 0.8 else 'Good' if psi > 0.6 else 'Fair',
            'components': {
                'water_efficiency': round(water_efficiency, 2),
                'fertilizer_efficiency': round(fertilizer_efficiency, 2),
                'yield_potential': round(yield_potential, 2),
                'sustainability': round(sustainability, 2)
            }
        }
    
    def generate_complete_advisory(self, crop, soil_data, climate_data, farm_size_ha=1.0):
        """Generate comprehensive crop advisory report"""
        print("\n" + "="*80)
        print(f"🌾 AgroMind+ Advisory Report for {crop}")
        print("="*80)
        
        # Soil analysis
        soil_analysis = self.analyze_soil_conditions(soil_data, crop)
        
        # Fertilizer plan
        fert_plan = self.generate_fertilizer_plan(soil_analysis, crop, farm_size_ha)
        
        # Irrigation plan
        irrigation_plan = self.generate_irrigation_plan(crop, climate_data)
        
        # Yield prediction
        yield_pred = self.predict_yield(crop, soil_data, climate_data, fertilizer_applied=True)
        
        # PSI calculation
        psi = self.calculate_psi(crop, soil_data, climate_data)
        
        report = {
            'crop': crop,
            'farm_size_ha': farm_size_ha,
            'timestamp': datetime.now().isoformat(),
            'soil_analysis': soil_analysis,
            'fertilizer_plan': fert_plan,
            'irrigation_plan': irrigation_plan,
            'yield_prediction': yield_pred,
            'psi': psi
        }
        
        # Print report
        self._print_advisory_report(report)
        
        return report
    
    def _print_advisory_report(self, report):
        """Print formatted advisory report"""
        print(f"\n📊 Farm Details:")
        print(f"   Crop: {report['crop']}")
        print(f"   Farm Size: {report['farm_size_ha']} hectares")
        
        print(f"\n🔬 Soil Analysis:")
        sa = report['soil_analysis']
        if sa['needs_amendment']:
            print(f"   ⚠️  Soil amendments needed")
            print(f"   N Deficit: {sa['N_deficit']:.1f} kg/ha")
            print(f"   P Deficit: {sa['P_deficit']:.1f} kg/ha")
            print(f"   K Deficit: {sa['K_deficit']:.1f} kg/ha")
        else:
            print(f"   ✓ Soil conditions are adequate")
        
        print(f"\n💊 Fertilizer Recommendations:")
        for fert in report['fertilizer_plan']['fertilizers']:
            print(f"   • {fert['name']}: {fert['quantity_kg']} kg")
            print(f"     Application: {fert['application']}")
            print(f"     Purpose: {fert['deficit_addressed']}")
        
        print(f"\n💧 Irrigation Plan:")
        irr = report['irrigation_plan']
        print(f"   Type: {irr['recommended_type']}")
        print(f"   Frequency: {irr['frequency']}")
        print(f"   Water Depth: {irr['water_depth']}")
        print(f"   Total Water Required: {irr['total_water_required_mm']} mm")
        if irr['adjustments']:
            print(f"   ⚠️  Adjustments:")
            for adj in irr['adjustments']:
                print(f"      - {adj['action']} ({adj['reason']})")
        
        print(f"\n📈 Yield Prediction:")
        yp = report['yield_prediction']
        print(f"   Expected Yield: {yp['predicted_yield_t_ha']} tons/hectare")
        print(f"   Confidence: {yp['confidence']}")
        
        print(f"\n🌱 Predictive Sustainability Index (PSI):")
        psi = report['psi']
        print(f"   PSI Score: {psi['psi_percentage']}% ({psi['rating']})")
        print(f"   Water Efficiency: {psi['components']['water_efficiency']}")
        print(f"   Fertilizer Efficiency: {psi['components']['fertilizer_efficiency']}")
        print(f"   Yield Potential: {psi['components']['yield_potential']}")

def main():
    """Demo of advisory system"""
    advisory = AdaptiveCropAdvisory()
    
    # Sample data
    soil_data = {
        'N': 110, 'P': 35, 'K': 38, 'pH': 6.8,
        'Moisture': 65
    }
    
    climate_data = {
        'Temperature': 28, 'Humidity': 75,
        'Rainfall': 85, 'Sunlight': 6.5
    }
    
    # Generate advisory for Aman Rice
    report = advisory.generate_complete_advisory(
        'Aman_Rice', soil_data, climate_data, farm_size_ha=2.0
    )
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
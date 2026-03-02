"""
AgroMind+ Integrated System
Combines LSTM prediction with adaptive advisory
"""

import numpy as np
import pandas as pd
from datetime import datetime
import joblib
from tensorflow import keras

from advisory_system import AdaptiveCropAdvisory

class AgroMindIntegratedSystem:
    def __init__(self, model_path='../models/agromind_lstm_model.h5'):
        """Initialize integrated system"""
        print("="*80)
        print("🌾 Initializing AgroMind+ Integrated System")
        print("="*80)
        
        # Load LSTM model
        try:
            from lstm_model import AttentionLayer
            self.model = keras.models.load_model(
                model_path,
                custom_objects={'AttentionLayer': AttentionLayer}
            )
            self.scaler = joblib.load('../models/feature_scaler.pkl')
            self.label_encoder = joblib.load('../models/label_encoder.pkl')
            print("✓ LSTM model loaded successfully")
        except Exception as e:
            print(f"⚠️  Model loading failed: {e}")
            print("   Please train the model first by running: python lstm_model.py")
            self.model = None
        
        # Initialize advisory system
        self.advisory = AdaptiveCropAdvisory()
        print("✓ Advisory system initialized")
        
        # Behavioral learning storage
        self.farmer_choices = []
        self.feedback_data = []
    
    def predict_top_crops(self, sequence_data, top_k=4):
        """Predict top-k crops using LSTM"""
        if self.model is None:
            return self._fallback_prediction()
        
        # Prepare sequence
        if len(sequence_data.shape) == 2:
            # Single sequence: (time_steps, features)
            sequence_scaled = self.scaler.transform(sequence_data.reshape(-1, 9))
            sequence_scaled = sequence_scaled.reshape(1, -1, 9)
        else:
            sequence_scaled = self.scaler.transform(sequence_data.reshape(-1, 9))
            sequence_scaled = sequence_scaled.reshape(1, sequence_data.shape[0], 9)
        
        # Predict
        predictions = self.model.predict(sequence_scaled, verbose=0)[0]
        
        # Get top-k
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_crops = self.label_encoder.inverse_transform(top_indices)
        top_probabilities = predictions[top_indices]
        
        # Calculate PSI for each crop
        current_conditions = {
            'N': sequence_data[-1, 0],
            'P': sequence_data[-1, 1],
            'K': sequence_data[-1, 2],
            'pH': sequence_data[-1, 3],
            'Temperature': sequence_data[-1, 4],
            'Humidity': sequence_data[-1, 5],
            'Moisture': sequence_data[-1, 6],
            'Rainfall': sequence_data[-1, 7],
            'Sunlight': sequence_data[-1, 8]
        }
        
        results = []
        for i, (crop, prob) in enumerate(zip(top_crops, top_probabilities)):
            psi = self.advisory.calculate_psi(crop, current_conditions, current_conditions)
            
            results.append({
                'rank': i + 1,
                'crop': crop,
                'suitability': round(float(prob), 4),
                'confidence': f"{prob*100:.2f}%",
                'psi_score': psi['psi_score'],
                'psi_rating': psi['rating'],
                'psi_percentage': psi['psi_percentage']
            })
        
        return results
    
    def _fallback_prediction(self):
        """Fallback when model not available"""
        crops = ['Aman_Rice', 'Wheat', 'Maize', 'Pulses']
        return [
            {
                'rank': i+1,
                'crop': crop,
                'suitability': round(0.9 - i*0.1, 2),
                'confidence': f"{(90-i*10):.0f}%",
                'psi_score': 0.75,
                'psi_rating': 'Good',
                'psi_percentage': 75.0
            }
            for i, crop in enumerate(crops)
        ]
    
    def farmer_interaction(self, recommendations, selected_rank=None):
        """Handle farmer's crop selection"""
        print("\n" + "="*80)
        print("📱 Farmer Interaction Interface")
        print("="*80)
        
        print("\n🌾 Top-4 Recommended Crops:")
        print("-" * 80)
        print(f"{'Rank':<6} {'Crop':<15} {'Suitability':<12} {'PSI':<10} {'Rating':<10}")
        print("-" * 80)
        
        for rec in recommendations:
            print(f"{rec['rank']:<6} {rec['crop']:<15} {rec['confidence']:<12} "
                  f"{rec['psi_percentage']:.1f}%{'':<5} {rec['psi_rating']:<10}")
        
        print("-" * 80)
        
        # Get farmer's choice
        if selected_rank is None:
            try:
                selected_rank = int(input("\n👨‍🌾 Select crop rank (1-4) or 0 for best recommendation: "))
                if selected_rank == 0:
                    selected_rank = 1
            except:
                selected_rank = 1
                print("   Using best recommendation (Rank 1)")
        
        selected_crop = recommendations[selected_rank - 1]
        
        print(f"\n✓ You selected: {selected_crop['crop']} (Rank {selected_rank})")
        
        if selected_rank > 1:
            print(f"\n💡 Note: This crop ranked #{selected_rank}. ")
            print(f"   We'll provide optimized recommendations to achieve best results!")
        
        return selected_crop
    
    def generate_adaptive_advisory(self, selected_crop, current_conditions, farm_size_ha=1.0):
        """Generate advisory for selected crop"""
        soil_data = {
            'N': current_conditions['N'],
            'P': current_conditions['P'],
            'K': current_conditions['K'],
            'pH': current_conditions['pH'],
            'Moisture': current_conditions['Moisture']
        }
        
        climate_data = {
            'Temperature': current_conditions['Temperature'],
            'Humidity': current_conditions['Humidity'],
            'Rainfall': current_conditions['Rainfall'],
            'Sunlight': current_conditions['Sunlight']
        }
        
        # Generate complete advisory
        advisory_report = self.advisory.generate_complete_advisory(
            selected_crop['crop'],
            soil_data,
            climate_data,
            farm_size_ha
        )
        
        # Generate explainable narrative
        narrative = self._generate_explainable_narrative(
            selected_crop, advisory_report
        )
        
        advisory_report['narrative'] = narrative
        
        return advisory_report
    
    def _generate_explainable_narrative(self, selected_crop, advisory_report):
        """Generate human-readable explanation (XCN)"""
        rank = selected_crop['rank']
        crop = selected_crop['crop']
        suitability = selected_crop['confidence']
        psi = advisory_report['psi']
        
        narrative = f"\n{'='*80}\n"
        narrative += f"📖 Explainable Crop Narrative (XCN)\n"
        narrative += f"{'='*80}\n\n"
        
        # Ranking explanation
        if rank == 1:
            narrative += f"✨ Excellent Choice! {crop} is our top recommendation.\n\n"
            narrative += f"Why this crop?\n"
            narrative += f"• Current environmental conditions perfectly match {crop} requirements\n"
            narrative += f"• Suitability score: {suitability}\n"
            narrative += f"• High sustainability rating: {psi['rating']} ({psi['psi_percentage']}%)\n"
        else:
            narrative += f"💡 You selected {crop} (Rank {rank}).\n\n"
            narrative += f"Understanding your choice:\n"
            narrative += f"• While ranked #{rank}, this crop can still perform well\n"
            narrative += f"• Current suitability: {suitability}\n"
            narrative += f"• With proper management, you can achieve excellent yields\n"
        
        narrative += f"\n📊 What makes this achievable?\n"
        
        # Soil conditions
        if advisory_report['soil_analysis']['needs_amendment']:
            narrative += f"• Soil nutrients need adjustment\n"
            narrative += f"  → We'll provide precise fertilizer recommendations\n"
        else:
            narrative += f"• Your soil conditions are already favorable\n"
        
        # Climate suitability
        yield_factors = advisory_report['yield_prediction']['factors']
        climate_score = yield_factors['climate_suitability']
        
        if climate_score > 0.8:
            narrative += f"• Climate conditions are excellent for {crop}\n"
        elif climate_score > 0.6:
            narrative += f"• Climate conditions are good, with minor adjustments needed\n"
        else:
            narrative += f"• Climate requires careful management\n"
            narrative += f"  → Follow our irrigation recommendations closely\n"
        
        # Yield potential
        pred_yield = advisory_report['yield_prediction']['predicted_yield_t_ha']
        narrative += f"\n🎯 Expected Outcome:\n"
        narrative += f"• Predicted yield: {pred_yield} tons/hectare\n"
        
        if rank > 1:
            potential_improvement = pred_yield * 0.12
            narrative += f"• With our fertilizer plan: +{potential_improvement:.2f} tons/ha boost possible\n"
            narrative += f"• Follow irrigation schedule to maximize results\n"
        
        # Sustainability
        narrative += f"\n🌱 Sustainability Impact:\n"
        narrative += f"• Water efficiency: {psi['components']['water_efficiency']*100:.0f}%\n"
        narrative += f"• Fertilizer efficiency: {psi['components']['fertilizer_efficiency']*100:.0f}%\n"
        narrative += f"• Overall PSI: {psi['rating']} - {psi['psi_percentage']}%\n"
        
        # Action items
        narrative += f"\n✅ Next Steps:\n"
        narrative += f"1. Review fertilizer recommendations below\n"
        narrative += f"2. Plan irrigation schedule\n"
        narrative += f"3. Monitor soil moisture regularly\n"
        narrative += f"4. Follow critical growth stage guidelines\n"
        
        narrative += f"\n{'='*80}\n"
        
        return narrative
    
    def record_farmer_choice(self, recommendations, selected_crop, advisory_report):
        """Record farmer's choice for behavioral learning"""
        choice_record = {
            'timestamp': datetime.now().isoformat(),
            'recommendations': recommendations,
            'selected_rank': selected_crop['rank'],
            'selected_crop': selected_crop['crop'],
            'advisory_report': advisory_report
        }
        
        self.farmer_choices.append(choice_record)
    
    def record_feedback(self, crop, actual_yield, satisfaction_score):
        """Record farmer feedback for continuous learning"""
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'crop': crop,
            'actual_yield': actual_yield,
            'satisfaction_score': satisfaction_score  # 1-5 scale
        }
        
        self.feedback_data.append(feedback)
        
        print(f"\n✓ Feedback recorded. Thank you!")
        print(f"   This helps us improve recommendations for your farm.")
    
    def run_complete_workflow(self, sequence_data, farm_size_ha=1.0, auto_select=None):
        """Run complete AgroMind+ workflow"""
        print("\n" + "="*80)
        print("🌾 AgroMind+ Complete Workflow")
        print("="*80)
        
        # Step 1: LSTM Prediction
        print("\n📊 Step 1: Analyzing environmental data with LSTM...")
        recommendations = self.predict_top_crops(sequence_data, top_k=4)
        
        # Step 2: Farmer Interaction
        print("\n👨‍🌾 Step 2: Farmer interaction...")
        selected_crop = self.farmer_interaction(recommendations, auto_select)
        
        # Step 3: Generate Advisory
        print("\n💡 Step 3: Generating adaptive advisory...")
        current_conditions = {
            'N': sequence_data[-1, 0],
            'P': sequence_data[-1, 1],
            'K': sequence_data[-1, 2],
            'pH': sequence_data[-1, 3],
            'Temperature': sequence_data[-1, 4],
            'Humidity': sequence_data[-1, 5],
            'Moisture': sequence_data[-1, 6],
            'Rainfall': sequence_data[-1, 7],
            'Sunlight': sequence_data[-1, 8]
        }
        
        advisory_report = self.generate_adaptive_advisory(
            selected_crop, current_conditions, farm_size_ha
        )
        
        # Step 4: Display Narrative
        print(advisory_report['narrative'])
        
        # Step 5: Record choice
        self.record_farmer_choice(recommendations, selected_crop, advisory_report)
        
        print("\n" + "="*80)
        print("✅ Workflow Complete!")
        print("="*80)
        
        return {
            'recommendations': recommendations,
            'selected_crop': selected_crop,
            'advisory_report': advisory_report
        }

def demo_system():
    """Demo of integrated system"""
    # Initialize system
    system = AgroMindIntegratedSystem()
    
    # Sample sequence data (4 weeks of measurements)
    sequence_data = np.array([
        [120, 40, 42, 6.8, 26, 72, 65, 80, 6.5],  # Week 1
        [115, 38, 40, 6.7, 27, 75, 68, 90, 6.0],  # Week 2
        [110, 36, 38, 6.8, 28, 78, 70, 95, 5.5],  # Week 3
        [108, 35, 37, 6.9, 29, 80, 72, 100, 5.8]  # Week 4
    ])
    
    print("\n📍 Current Farm Conditions:")
    print(f"   N: {sequence_data[-1, 0]:.1f} kg/ha")
    print(f"   P: {sequence_data[-1, 1]:.1f} kg/ha")
    print(f"   K: {sequence_data[-1, 2]:.1f} kg/ha")
    print(f"   pH: {sequence_data[-1, 3]:.1f}")
    print(f"   Temperature: {sequence_data[-1, 4]:.1f}°C")
    print(f"   Humidity: {sequence_data[-1, 5]:.1f}%")
    print(f"   Rainfall: {sequence_data[-1, 7]:.1f} mm")
    
    # Run workflow with auto-selection (for demo)
    result = system.run_complete_workflow(
        sequence_data,
        farm_size_ha=2.0,
        auto_select=1  # Auto-select rank 1 for demo
    )
    
    return result

if __name__ == "__main__":
    demo_system()
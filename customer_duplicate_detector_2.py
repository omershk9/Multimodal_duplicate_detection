# customer_duplicate_detection.py
import pandas as pd
import numpy as np
import ast
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Required imports
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import re

class CustomerTextProcessor:
    def __init__(self, model_name='distilbert-base-uncased'):
        print("Loading BERT model for text processing...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.pca = PCA(n_components=64)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        
    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def get_embeddings(self, texts, batch_size=16):
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                  max_length=128, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def process_customer_text(self, df):
        """Combine name and city for text analysis"""
        combined_text = []
        for _, row in df.iterrows():
            name_clean = self.clean_text(row['name'])
            city_clean = self.clean_text(row['city'])
            combined = f"{name_clean} {city_clean}".strip()
            combined_text.append(combined if combined else "unknown")
        
        print(f"Generating embeddings for {len(combined_text)} customer records...")
        embeddings = self.get_embeddings(combined_text)
        print("Applying PCA dimensionality reduction...")
        reduced_embeddings = self.pca.fit_transform(embeddings)
        return reduced_embeddings, combined_text

class CustomerBehaviorProcessor:
    def parse_login_times(self, login_times_str):
        """Parse the login_times string into datetime objects"""
        try:
            # Handle string representation of list
            if isinstance(login_times_str, str):
                login_times_list = ast.literal_eval(login_times_str)
            else:
                login_times_list = login_times_str
            
            timestamps = []
            for ts in login_times_list:
                try:
                    timestamps.append(pd.to_datetime(ts))
                except:
                    continue
            return timestamps
        except:
            return []
    
    def extract_behavior_features(self, df):
        """Extract behavioral patterns from login times"""
        features = []
        
        for _, row in df.iterrows():
            timestamps = self.parse_login_times(row['login_times'])
            
            if len(timestamps) == 0:
                # Default features for customers with no login data
                feature_vector = [0, 0, 0, 12, 0.3, 1]
            else:
                timestamps.sort()
                
                # Calculate time differences between logins
                if len(timestamps) > 1:
                    time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() / 3600 
                                 for i in range(len(timestamps)-1)]
                    avg_gap = np.mean(time_diffs)
                    std_gap = np.std(time_diffs)
                else:
                    avg_gap = 0
                    std_gap = 0
                
                # Extract hour patterns
                hours = [ts.hour for ts in timestamps]
                days = [ts.weekday() for ts in timestamps]  # 0=Monday, 6=Sunday
                
                feature_vector = [
                    len(timestamps),  # login_count
                    avg_gap,  # avg_time_between_logins (hours)
                    std_gap,  # std_time_between_logins
                    np.bincount(hours).argmax() if hours else 12,  # most_common_hour
                    np.sum([d >= 5 for d in days]) / len(days) if days else 0.3,  # weekend_ratio
                    np.var(hours) if len(hours) > 1 else 1  # hour_variance
                ]
            
            features.append(feature_vector)
        
        return np.array(features)

class CustomerDeviceProcessor:
    def __init__(self):
        self.label_encoders = {}
        
    def process_device_features(self, df):
        """Process browser and OS information"""
        device_features = []
        
        # Process browser
        if 'browser_encoded' not in self.label_encoders:
            self.label_encoders['browser_encoded'] = LabelEncoder()
        browser_encoded = self.label_encoders['browser_encoded'].fit_transform(
            df['browser'].fillna('unknown')
        )
        
        # Process OS
        if 'os_encoded' not in self.label_encoders:
            self.label_encoders['os_encoded'] = LabelEncoder()
        os_encoded = self.label_encoders['os_encoded'].fit_transform(
            df['os'].fillna('unknown')
        )
        
        # Create feature matrix
        for i in range(len(df)):
            feature_vector = [
                browser_encoded[i] / (browser_encoded.max() + 1),  # normalized browser
                os_encoded[i] / (os_encoded.max() + 1),  # normalized OS
            ]
            device_features.append(feature_vector)
        
        return np.array(device_features)

class CustomerDuplicateDetector:
    def __init__(self, eps=0.3, min_samples=2):
        self.text_processor = CustomerTextProcessor()
        self.behavior_processor = CustomerBehaviorProcessor()
        self.device_processor = CustomerDeviceProcessor()
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        
    def calculate_similarity_score(self, idx1, idx2, text_clusters, behavior_features, device_features):
        """Calculate multi-modal similarity score between two customers"""
        
        # Text similarity (1.0 if same cluster, 0.0 otherwise)
        text_sim = 1.0 if (text_clusters[idx1] == text_clusters[idx2] and text_clusters[idx1] != -1) else 0.0
        
        # Behavior similarity (cosine similarity)
        behav1, behav2 = behavior_features[idx1], behavior_features[idx2]
        behav_sim = np.dot(behav1, behav2) / (np.linalg.norm(behav1) * np.linalg.norm(behav2) + 1e-8)
        behav_sim = max(0, behav_sim)  # Ensure non-negative
        
        # Device similarity (cosine similarity)
        device1, device2 = device_features[idx1], device_features[idx2]
        device_sim = np.dot(device1, device2) / (np.linalg.norm(device1) * np.linalg.norm(device2) + 1e-8)
        device_sim = max(0, device_sim)
        
        # Weighted fusion (LANISTR-inspired)
        weights = {'text': 0.5, 'behavior': 0.3, 'device': 0.2}
        final_score = (weights['text'] * text_sim + 
                      weights['behavior'] * behav_sim + 
                      weights['device'] * device_sim)
        
        return final_score, {
            'text_similarity': text_sim,
            'behavior_similarity': behav_sim,
            'device_similarity': device_sim
        }
    
    def detect_duplicates(self, df, confidence_threshold=0.4):
        """Main function to detect duplicate customers"""
        print("=== Customer Duplicate Detection System ===")
        print(f"Processing {len(df)} customer records...")
        
        # Step 1: Process text features (name + city)
        print("\n1. Processing text features...")
        text_embeddings, combined_texts = self.text_processor.process_customer_text(df)
        
        # Step 2: Process behavior features (login patterns)
        print("2. Processing behavior features...")
        behavior_features = self.behavior_processor.extract_behavior_features(df)
        
        # Step 3: Process device features (browser + OS)
        print("3. Processing device features...")
        device_features = self.device_processor.process_device_features(df)
        
        # Step 4: Run DBSCAN clustering on text embeddings
        print("4. Running DBSCAN clustering...")
        cluster_labels = self.dbscan.fit_predict(text_embeddings)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        print(f"   Found {n_clusters} clusters with {n_noise} noise points")
        
        # Step 5: Identify potential duplicates
        print("5. Identifying potential duplicate pairs...")
        duplicate_results = []
        
        # Check all pairs within each cluster
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Skip noise points
                continue
                
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) > 1:
                for i in range(len(cluster_indices)):
                    for j in range(i + 1, len(cluster_indices)):
                        idx1, idx2 = cluster_indices[i], cluster_indices[j]
                        
                        # Calculate similarity score
                        final_score, detailed_scores = self.calculate_similarity_score(
                            idx1, idx2, cluster_labels, behavior_features, device_features
                        )
                        
                        if final_score >= confidence_threshold:
                            duplicate_results.append({
                                'customer_1_id': df.iloc[idx1]['customer_id'],
                                'customer_1_name': df.iloc[idx1]['name'],
                                'customer_1_city': df.iloc[idx1]['city'],
                                'customer_1_browser': df.iloc[idx1]['browser'],
                                'customer_1_os': df.iloc[idx1]['os'],
                                'customer_2_id': df.iloc[idx2]['customer_id'],
                                'customer_2_name': df.iloc[idx2]['name'],
                                'customer_2_city': df.iloc[idx2]['city'],
                                'customer_2_browser': df.iloc[idx2]['browser'],
                                'customer_2_os': df.iloc[idx2]['os'],
                                'confidence_score': final_score,
                                'text_similarity': detailed_scores['text_similarity'],
                                'behavior_similarity': detailed_scores['behavior_similarity'],
                                'device_similarity': detailed_scores['device_similarity'],
                                'cluster_id': cluster_id
                            })
        
        # Sort by confidence score
        duplicate_results.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        # Generate summary
        summary = {
            'total_customers': len(df),
            'potential_duplicates_found': len(duplicate_results),
            'clusters_formed': n_clusters,
            'noise_points': n_noise,
            'high_confidence_duplicates': len([r for r in duplicate_results if r['confidence_score'] > 0.7]),
            'medium_confidence_duplicates': len([r for r in duplicate_results if 0.5 <= r['confidence_score'] <= 0.7]),
            'low_confidence_duplicates': len([r for r in duplicate_results if r['confidence_score'] < 0.5])
        }
        
        return duplicate_results, summary
    
    def print_results(self, duplicate_results, summary):
        """Print formatted results"""
        print("\n" + "="*60)
        print("DUPLICATE DETECTION RESULTS")
        print("="*60)
        
        print(f"Total customers processed: {summary['total_customers']}")
        print(f"Potential duplicate pairs found: {summary['potential_duplicates_found']}")
        print(f"Clusters formed: {summary['clusters_formed']}")
        print(f"Noise points: {summary['noise_points']}")
        print(f"High confidence (>0.7): {summary['high_confidence_duplicates']}")
        print(f"Medium confidence (0.5-0.7): {summary['medium_confidence_duplicates']}")
        print(f"Low confidence (<0.5): {summary['low_confidence_duplicates']}")
        
        if duplicate_results:
            print(f"\nTOP {min(10, len(duplicate_results))} DUPLICATE PAIRS:")
            print("-" * 120)
            
            for i, result in enumerate(duplicate_results[:10]):
                print(f"\nPair {i+1} (Confidence: {result['confidence_score']:.3f}):")
                print(f"  Customer 1: {result['customer_1_name']} | {result['customer_1_city']} | {result['customer_1_browser']}/{result['customer_1_os']}")
                print(f"  Customer 2: {result['customer_2_name']} | {result['customer_2_city']} | {result['customer_2_browser']}/{result['customer_2_os']}")
                print(f"  Similarities - Text: {result['text_similarity']:.3f}, Behavior: {result['behavior_similarity']:.3f}, Device: {result['device_similarity']:.3f}")
                print(f"  Customer IDs: {result['customer_1_id'][:8]}... | {result['customer_2_id'][:8]}...")
        else:
            print("\nNo duplicate pairs found above the confidence threshold.")

# Main execution function
def run_duplicate_detection(csv_file_path, confidence_threshold=0.4, eps=0.3, min_samples=2):
    """
    Run duplicate detection on your CSV file
    
    Parameters:
    - csv_file_path: path to your CSV file
    - confidence_threshold: minimum confidence score to consider as duplicate (0.0-1.0)
    - eps: DBSCAN epsilon parameter (distance threshold)
    - min_samples: minimum samples to form a cluster
    """
    
    # Load the CSV
    print(f"Loading data from {csv_file_path}...")
    df = pd.read_csv(csv_file_path)
    
    # Verify expected columns
    expected_columns = ['customer_id', 'name', 'city', 'browser', 'os', 'login_times']
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return None, None
    
    print(f"Data loaded successfully: {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    
    # Initialize detector
    detector = CustomerDuplicateDetector(eps=eps, min_samples=min_samples)
    
    # Run detection
    duplicate_results, summary = detector.detect_duplicates(df, confidence_threshold)
    
    # Print results
    detector.print_results(duplicate_results, summary)
    
    # Save results to CSV
    if duplicate_results:
        results_df = pd.DataFrame(duplicate_results)
        output_file = csv_file_path.replace('.csv', '_duplicates.csv')
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    
    return duplicate_results, summary

# Example usage:
if __name__ == "__main__":
    # Replace 'your_file.csv' with the path to your CSV file
    csv_file_path = "Simulated_CRM_Dataset.csv"
    
    # Run duplicate detection
    duplicates, summary = run_duplicate_detection(
        csv_file_path=csv_file_path,
        confidence_threshold=0.4,  # Adjust this threshold as needed
        eps=0.3,  # DBSCAN parameter - smaller = stricter clustering
        min_samples=2  # Minimum customers to form a cluster
    )

# Instructions for running:
"""
INSTRUCTIONS FOR RUNNING ON YOUR CSV:

1. Install required dependencies:
   pip install pandas numpy transformers torch scikit-learn

2. Save this code as 'customer_duplicate_detection.py'

3. Update the csv_file_path variable to point to your CSV file

4. Run the script:
   python customer_duplicate_detection.py

5. Adjust parameters as needed:
   - confidence_threshold: Higher = fewer, more confident duplicates
   - eps: DBSCAN clustering sensitivity (0.2-0.5 typically work well)
   - min_samples: Minimum customers to form a cluster

6. The script will output:
   - Console results showing duplicate pairs
   - A CSV file with detailed duplicate results
   - Summary statistics

EXPECTED OUTPUT:
- List of potential duplicate customer pairs
- Confidence scores for each pair
- Individual similarity scores (text, behavior, device)
- Customer details for manual verification
"""

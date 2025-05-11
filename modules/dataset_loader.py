import kagglehub
import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def download_ielts_dataset():
    """
    Download the IELTS Writing Scored Essays dataset from Kaggle
    
    Returns:
        str: Path to the downloaded dataset
    """
    try:
        logger.info("Downloading IELTS Writing Scored Essays dataset...")
        path = kagglehub.dataset_download("mazlumi/ielts-writing-scored-essays-dataset")
        logger.info(f"Dataset downloaded to: {path}")
        return path
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        raise

def load_ielts_dataset(path=None):
    """
    Load the IELTS Writing Scored Essays dataset
    
    Args:
        path (str, optional): Path to the dataset. If None, downloads the dataset.
        
    Returns:
        pandas.DataFrame: DataFrame containing the dataset
    """
    try:
        if path is None:
            path = download_ielts_dataset()
            
        # Find the CSV file in the directory
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.endswith('.csv'):
                    csv_path = os.path.join(path, file)
                    break
            else:
                raise FileNotFoundError(f"No CSV file found in {path}")
        else:
            csv_path = path
            
        logger.info(f"Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Dataset loaded with {len(df)} records")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise
        
def get_sample_essays_from_dataset(n=5):
    """
    Get sample essays from the IELTS dataset
    
    Args:
        n (int, optional): Number of sample essays to return. Defaults to 5.
        
    Returns:
        list: List of dictionaries containing sample essays with their scores
    """
    try:
        df = load_ielts_dataset()
        
        # Select a diverse set of essays with different scores
        samples = []
        
        # Ensure we have essays of different score bands if possible
        if 'Score' in df.columns:
            for score in sorted(df['Score'].unique(), reverse=True)[:n]:
                essay = df[df['Score'] == score].iloc[0]
                samples.append({
                    'title': f"IELTS Essay (Band {essay['Score']})",
                    'text': essay['Essay'],
                    'score': essay['Score']
                })
        
        # If we don't have enough samples, add more
        if len(samples) < n:
            additional = df.sample(n - len(samples))
            for _, essay in additional.iterrows():
                samples.append({
                    'title': f"IELTS Essay (Band {essay['Score']})" if 'Score' in df.columns else "IELTS Essay Sample",
                    'text': essay['Essay'],
                    'score': essay['Score'] if 'Score' in df.columns else None
                })
        
        return samples[:n]
    except Exception as e:
        logger.error(f"Error getting sample essays: {str(e)}")
        # Return some default samples if we can't load from dataset
        return [
            {
                'title': "Sample Essay",
                'text': "This is a sample essay. Please load your own essay or download the dataset to see real IELTS essays."
            }
        ]
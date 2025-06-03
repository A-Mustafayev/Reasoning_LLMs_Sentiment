import pandas as pd
import openai
import time
import logging
import os
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sentiment_analysis.log"),
        logging.StreamHandler()
    ]
)

# Load your DeepSeek API key
client = openai.OpenAI(
    api_key="Your_API_Key",
    base_url="https://api.deepseek.com"
)

# File paths
input_file = 'Your_Excel_Input_Filepath'
output_file = 'Your_Excel_Output_Filepath'
checkpoint_file = 'Your_Checkpoint_Filepath'

# Function to get sentiment for a batch of reviews
def get_sentiment_batch(review_ids, reviews, batch_size=50, max_retries=5):
    results = []
    total_reviews = len(reviews)
    
    # Process in smaller batches
    for i in range(0, total_reviews, batch_size):
        end_idx = min(i + batch_size, total_reviews)
        batch_ids = review_ids[i:end_idx]
        batch_reviews = reviews[i:end_idx]
        
        # Prepare the prompt
        reviews_text = '\n'.join(f'{review_id}: "{review}"' for review_id, review in zip(batch_ids, batch_reviews))
        prompt = f"""Classify the sentiment in these reviews as positive, negative, or neutral". Return ONLY the review ID followed by a colon and then the sentiment (either 'positive', 'negative', or 'neutral'). 
No other text, explanations, or formatting should be included. Format each review on a new line like this: 'ID: positive' or 'ID: negative' or 'ID: neutral'


{reviews_text}"""
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Call the DeepSeek API
                logging.info(f"Processing batch {i//batch_size + 1}, reviews {i+1} to {end_idx} (total in batch: {len(batch_ids)})")
                
                # Create the message structure
                messages = [{"role": "user", "content": prompt}]
                
                # Call DeepSeek Reasoner API
                response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=messages,
                )
                
                # Extract both reasoning content and final answer
                reasoning_content = response.choices[0].message.reasoning_content
                response_text = response.choices[0].message.content
                
                # Log the reasoning for debugging if needed
                logging.debug(f"Reasoning content: {reasoning_content}")
                
                # Process response text
                response_lines = response_text.strip().split("\n")
                
                # Log the response stats
                logging.info(f"Received {len(response_lines)} results from API for batch of {len(batch_ids)} reviews")
                
                # Validate response format and extract sentiments
                batch_results = []
                for line in response_lines:
                    try:
                        if ":" in line:
                            parts = line.split(":", 1)
                            review_id = parts[0].strip()
                            sentiment = parts[1].strip().lower()
                            
                            if sentiment in ['positive', 'negative', 'neutral']:
                                batch_results.append((review_id, sentiment))
                            else:
                                logging.warning(f"Invalid sentiment value: {sentiment} for review {review_id}")
                        else:
                            logging.warning(f"Invalid response format: {line}")
                    except Exception as e:
                        logging.error(f"Error parsing line: {line}. Error: {str(e)}")
                
                # Check if we got results for all reviews in this batch
                if len(batch_results) < len(batch_ids):
                    logging.warning(f"Got {len(batch_results)} results for {len(batch_ids)} reviews in batch")
                
                results.extend(batch_results)
                break
            
            except Exception as e:
                retry_count += 1
                if "Rate limit exceeded" in str(e):
                    wait_time = 20 * retry_count  # Increase wait time with each retry
                    logging.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds. Retry {retry_count}/{max_retries}")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Error: {str(e)}. Retrying in 5 seconds. Retry {retry_count}/{max_retries}")
                    time.sleep(5)
                
                if retry_count >= max_retries:
                    logging.error(f"Max retries reached for batch {i//batch_size + 1}. Moving to next batch.")
        
        # Save checkpoint after each batch
        save_checkpoint(results)
        
        # Sleep briefly between batches to avoid rate limits
        time.sleep(1)
    
    return results

# Function to save checkpoint
def save_checkpoint(results):
    temp_df = pd.DataFrame(results, columns=["Review_ID", "Sentiment"])
    temp_df.to_pickle(checkpoint_file)
    logging.info(f"Checkpoint saved with {len(results)} results")

# Function to load checkpoint if exists
def load_checkpoint():
    if os.path.exists(checkpoint_file):
        try:
            checkpoint_df = pd.read_pickle(checkpoint_file)
            logging.info(f"Loaded checkpoint with {len(checkpoint_df)} results")
            return checkpoint_df["Review_ID"].tolist(), checkpoint_df["Sentiment"].tolist()
        except Exception as e:
            logging.error(f"Error loading checkpoint: {str(e)}")
    return [], []

# Handle streaming response (alternative approach)
def process_streaming_response(review_ids, reviews, batch_size=50, max_retries=5):
    results = []
    total_reviews = len(reviews)
    
    for i in range(0, total_reviews, batch_size):
        end_idx = min(i + batch_size, total_reviews)
        batch_ids = review_ids[i:end_idx]
        batch_reviews = reviews[i:end_idx]
        
        reviews_text = '\n'.join(f'{review_id}: "{review}"' for review_id, review in zip(batch_ids, batch_reviews))
        prompt = f"""Classify the sentiment in these reviews as positive, negative, or neutral". Return ONLY the review ID followed by a colon and then the sentiment (either 'positive', 'negative', or 'neutral'). 
No other text, explanations, or formatting should be included. Format each review on a new line like this: 'ID: positive' or 'ID: negative' or 'ID: neutral'

{reviews_text}"""
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                logging.info(f"Processing batch {i//batch_size + 1}, reviews {i+1} to {end_idx}")
                
                messages = [{"role": "user", "content": prompt}]
                response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=messages,
                    stream=True
                )
                
                reasoning_content = ""
                content = ""
                
                for chunk in response:
                    if chunk.choices[0].delta.reasoning_content:
                        reasoning_content += chunk.choices[0].delta.reasoning_content
                    else:
                        content += chunk.choices[0].delta.content
                
                # Process the final content
                response_lines = content.strip().split("\n")
                logging.info(f"Received {len(response_lines)} results from API")
                
                batch_results = []
                for line in response_lines:
                    try:
                        if ":" in line:
                            parts = line.split(":", 1)
                            review_id = parts[0].strip()
                            sentiment = parts[1].strip().lower()
                            
                            if sentiment in ['positive', 'negative', 'neutral']:
                                batch_results.append((review_id, sentiment))
                    except Exception as e:
                        logging.error(f"Error parsing line: {line}. Error: {str(e)}")
                
                results.extend(batch_results)
                break
                
            except Exception as e:
                retry_count += 1
                logging.error(f"Error: {str(e)}. Retrying {retry_count}/{max_retries}")
                time.sleep(5)
                
                if retry_count >= max_retries:
                    logging.error(f"Max retries reached for batch {i//batch_size + 1}")
        
        save_checkpoint(results)
        time.sleep(1)
    
    return results

# Main execution
def main():
    logging.info("Starting sentiment analysis process with DeepSeek Reasoner")
    
    # Load the dataset
    logging.info(f"Loading dataset from {input_file}")
    df = pd.read_excel(input_file)
    df['Review_ID'] = df['Review_ID'].astype(str)
    
    total_reviews = len(df)
    logging.info(f"Loaded {total_reviews} reviews for analysis")
    
    # Check for existing checkpoint
    existing_ids, existing_sentiments = load_checkpoint()
    sentiment_results = list(zip(existing_ids, existing_sentiments))
    
    if sentiment_results:
        # Create a set of already processed review IDs for quick lookup
        processed_ids = set(existing_ids)
        
        # Filter out already processed reviews
        to_process_df = df[~df['Review_ID'].isin(processed_ids)]
        logging.info(f"Found {len(sentiment_results)} already processed reviews. {len(to_process_df)} reviews remaining.")
    else:
        to_process_df = df
        logging.info(f"No checkpoint found. Processing all {total_reviews} reviews.")
    
    # If there are reviews left to process
    if len(to_process_df) > 0:
        review_ids = to_process_df['Review_ID'].tolist()
        reviews = to_process_df['Review'].tolist()
        
        # Choose the processing method - either with or without streaming
        use_streaming = False  # Set to True if you want to use streaming
        
        if use_streaming:
            new_results = process_streaming_response(review_ids, reviews, batch_size=50)
        else:
            new_results = get_sentiment_batch(review_ids, reviews, batch_size=50)
        
        # Combine with existing results
        sentiment_results.extend(new_results)
    
    # Create final DataFrame
    logging.info(f"Analysis complete. Got sentiment for {len(sentiment_results)} out of {total_reviews} reviews")
    df_results = pd.DataFrame(sentiment_results, columns=["Review_ID", "Sentiment"])
    
    # Save to Excel
    df_results.to_excel(output_file, index=False)
    logging.info(f"Results saved to {output_file}")
    
    # Check if any reviews were missed
    if len(df_results) < total_reviews:
        missing_count = total_reviews - len(df_results)
        logging.warning(f"⚠️ {missing_count} reviews did not receive sentiment analysis")
        
        # Find which reviews were missed
        processed_ids = set(df_results["Review_ID"].tolist())
        all_ids = set(df["Review_ID"].tolist())
        missed_ids = all_ids - processed_ids
        
        logging.warning(f"First 10 missed review IDs: {list(missed_ids)[:10]}")

if __name__ == "__main__":
    main()
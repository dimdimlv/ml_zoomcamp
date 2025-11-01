import pickle


def load_pipeline(filepath):
    """Load the pipeline from a pickle file."""
    with open(filepath, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline


def score_record(pipeline, record):
    """Score a single record using the pipeline."""
    # The pipeline expects a list of dictionaries
    probability = pipeline.predict_proba([record])[0, 1]
    return probability


def main():
    # Load the pipeline
    pipeline = load_pipeline('pipeline_v1.bin')
    
    # The record to score
    record = {
        "lead_source": "paid_ads",
        "number_of_courses_viewed": 2,
        "annual_income": 79276.0
    }
    
    # Get the probability
    probability = score_record(pipeline, record)
    
    print(f"Record: {record}")
    print(f"Probability of conversion: {probability:.3f}")


if __name__ == "__main__":
    main()

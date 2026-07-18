from google import genai
import re

# Initialize OpenAI client
client = genai.Client()


def analyze_sentiment(review):
    """
    Analyze the sentiment of a movie review using structured output.
    Returns a dictionary with 'thought' and 'sentiment' keys.
    """
    # TODO: Create a prompt that:
    # 1. Asks for sentiment analysis
    # 2. Specifies the required output format
    #       thought: [analysis]
    #       sentiment: [positive/negative]
    # 3. Includes the review text
    prompt = f"""
        We would like to analyze the movies based on the review provided and extract the thoughts and sentiments here.

        Input movie review text to use to analyze: {review}

        Response should be:
        sentiment: negative or positive
        thought: analyze the review to determine whether is it positive or negative
       
    """
    # Response:
    # Response should in the form of 2 json lists with thoughts values and sentiment values
    response = client.interactions.create(
        model="gemini-3.1-flash-lite",
        input=prompt
    )

    print("Response: =====> ", response.output_text)

    result = response.output_text.splitlines()

    # content = response.choices[0].message.content[0]
    # TODO: Parse the response to extract thought and sentiment
    # The response should be in the format:
    # thought: [analysis]
    # sentiment: [positive/negative]
    result = {
        # TODO: Extract thought
        "thought": re.search(r"thought:\s*(.+)", response.output_text).group(1),
        # TODO: Extract sentiment
        "sentiment": re.search(r"sentiment:\s*(\w+)", response.output_text).group(1)
    }

    return result


def main():
    # Test cases
    reviews = [
        "This film shouldn't work at all. It doesn't have much of a story and the whole dial up internet thing is incredibly dated. However Hanks and Ryan sell it beautifully.",
        "The movie was terrible. The acting was wooden, the plot made no sense, and I want my two hours back.",
        "An absolute masterpiece! The cinematography was stunning, the acting was superb, and the story kept me engaged from start to finish."
    ]

    # Test each review
    for i, review in enumerate(reviews, 1):
        result = analyze_sentiment(review)
        print(f"\nReview {i}:")
        print(f"Thought: {result['thought']}")
        print(f"Sentiment: {result['sentiment']}")


if __name__ == "__main__":
    main()

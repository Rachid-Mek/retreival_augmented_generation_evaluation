import re

def extract_text(results):
  """
  Extracts informative text from machine responses, handling specific formatting using regular expressions.

  Args:
      results: A list of strings containing machine responses.

  Returns:
      A list of strings containing the extracted text from each response.
  """
  pattern = r"(According to|Based on)\s+.*?,"  # Matches "According to" or "Based on" followed by anything until a comma (,)
  extracted_text = []
  for result in results:
    # Substitute the matched pattern with an empty string
    cleaned_text = re.sub(pattern, "", result, flags=re.IGNORECASE)
    extracted_text.append(cleaned_text.strip())  # Remove leading/trailing whitespaces

  return extracted_text